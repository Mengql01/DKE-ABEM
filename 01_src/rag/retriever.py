"""
RAG Retriever for TRNSYS DCK Generator.

Provides multi-index retrieval supporting the "copy when possible" and "copy first, then modify" principles:
- copy when possible: Retrieve similar system designs for reference
- copy first, then modify: Retrieve format templates to adapt

Uses the official LlamaIndex retriever API - NO custom reranking.

Used by:
- B2: RAG method (single LLM + RAG)
- B3: MultiAgent_RAG method (multi-agent + RAG)

Note: Proposed: MultiAgent_Memory method does NOT use RAG.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from llama_index.core import VectorStoreIndex
    from ..core.config import Settings

logger = logging.getLogger(__name__)


# === Data Models ===

class RetrievalResult(BaseModel):
    """Single retrieval result."""

    content: str = Field(description="Retrieved content text")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")
    score: float = Field(default=0.0, description="Relevance score")
    source_index: str = Field(default="", description="Source index name")
    node_id: str = Field(default="", description="Node ID in the index")


class RetrievalResponse(BaseModel):
    """Retrieval response."""

    results: List[RetrievalResult] = Field(default_factory=list, description="List of results")
    query: str = Field(default="", description="Original query")
    index_used: str = Field(default="", description="Index used for retrieval")
    total_found: int = Field(default=0, description="Total results found")
    retrieval_type: str = Field(default="", description="Type of retrieval performed")


# === Main Retriever Class ===

class RAGRetriever:
    """
    Multi-index RAG retriever.

    Provides retrieval methods for TRNSYS DCK generation:
    - System-level retrieval (similar systems)
    - Component-level retrieval (TMF specs)
    - Format-level retrieval (DCK templates)

    Uses LlamaIndex official retriever API without custom reranking.
    """

    def __init__(self, settings: Optional[Settings] = None):
        """
        Initialize RAG Retriever.

        Args:
            settings: Application settings. If None, loads from get_settings().
        """
        if settings is None:
            from ..core.config import get_settings
            settings = get_settings()

        self.settings = settings
        self._indexer = None

    @property
    def indexer(self):
        """Get or create RAG Indexer."""
        if self._indexer is None:
            from .indexer import get_rag_indexer
            self._indexer = get_rag_indexer()
        return self._indexer

    # === Query Preprocessing Methods ===

    @property
    def _query_llm(self):
        """Get or create a lightweight LLM instance for query preprocessing."""
        if not hasattr(self, '_query_llm_instance') or self._query_llm_instance is None:
            try:
                from ..core.llm_client import create_llm_client
                self._query_llm_instance = create_llm_client(
                    self.settings, temperature_override=0.0
                )
            except Exception as e:
                logger.warning(f"Failed to create query preprocessing LLM: {e}")
                self._query_llm_instance = None
        return self._query_llm_instance

    def _extract_english_keywords(self, text: str) -> str:
        """Extract 3-6 English keywords/phrases from any text for vector retrieval.

        Always uses LLM to distill the input into concise English search terms,
        regardless of input length or language. This ensures optimal vector
        similarity matching against the English-language RAG indexes.

        Args:
            text: Input text (any language, any length).

        Returns:
            Comma-separated English keywords/phrases (e.g.,
            "single zone building, PID controller, heating cooling").
        """
        llm = self._query_llm
        if llm is None:
            return self._fallback_extract_keywords(text)

        try:
            from langchain_core.messages import HumanMessage, SystemMessage

            messages = [
                SystemMessage(content=(
                    "You are a keyword extractor for TRNSYS energy simulation system retrieval. "
                    "Extract exactly 3-6 English keywords or short phrases that best describe "
                    "the TRNSYS system being requested. Focus on: system type, key components, "
                    "control strategies, and building type. "
                    "Output ONLY the keywords separated by commas, nothing else. "
                    "Example output: single zone building, PID controller, heating cooling, lumped capacitance"
                )),
                HumanMessage(content=text),
            ]
            response = llm.invoke(messages)
            keywords = response.content.strip()

            if keywords:
                logger.info(f"Extracted keywords: {keywords}")
                return keywords

        except Exception as e:
            logger.warning(f"LLM keyword extraction failed: {e}")

        return self._fallback_extract_keywords(text)

    def _translate_to_english(self, text: str) -> str:
        """Translate non-English text to concise English for vector retrieval.

        Detects Chinese characters and translates to English if found.
        For already-English text, returns as-is.

        Args:
            text: Input text (any language).

        Returns:
            English text suitable for vector retrieval.
        """
        if not text or not re.search(r'[\u4e00-\u9fff]', text):
            return text

        llm = self._query_llm
        if llm is None:
            return text

        try:
            from langchain_core.messages import HumanMessage, SystemMessage

            messages = [
                SystemMessage(content=(
                    "Translate the following text to concise English (max 50 words). "
                    "Focus on technical TRNSYS/HVAC terminology. "
                    "Output ONLY the translation, nothing else."
                )),
                HumanMessage(content=text),
            ]
            response = llm.invoke(messages)
            translated = response.content.strip()

            if translated:
                logger.info(f"Translated query: '{text[:50]}...' -> '{translated}'")
                return translated

        except Exception as e:
            logger.warning(f"LLM translation failed: {e}")

        return text

    def _fallback_extract_keywords(self, text: str) -> str:
        """Fallback keyword extraction using regex when LLM is unavailable.

        Extracts Type numbers and English technical terms from the text.

        Args:
            text: Input text.

        Returns:
            Space-separated keywords.
        """
        keywords = []

        # Extract Type numbers (e.g., Type15, Type56, Type927)
        type_matches = re.findall(r'Type\s*(\d+[\w-]*)', text, re.IGNORECASE)
        for t in type_matches[:6]:
            keywords.append(f"Type{t}")

        # Extract English words (3+ chars, likely technical terms)
        english_words = re.findall(r'[A-Za-z]{3,}', text)
        # Filter common stop words
        stop_words = {
            'the', 'and', 'for', 'with', 'from', 'that', 'this', 'are', 'was',
            'will', 'has', 'have', 'been', 'not', 'but', 'can', 'all', 'you',
            'your', 'use', 'using', 'used', 'please', 'based', 'following',
            'should', 'would', 'could', 'also', 'each', 'which', 'when',
            'example', 'include', 'including',
        }
        for word in english_words:
            if word.lower() not in stop_words and word not in keywords:
                keywords.append(word)
            if len(keywords) >= 6:
                break

        result = ", ".join(keywords) if keywords else text[:200]
        logger.info(f"Fallback keywords: {result}")
        return result

    # === Core Retrieval Methods ===

    def retrieve(
        self,
        query: str,
        index_name: Optional[str] = None,
        top_k: int = 5,
    ) -> RetrievalResponse:
        """
        General retrieval method.

        Args:
            query: Search query.
            index_name: Specific index to search. If None, searches all indexes.
            top_k: Number of results to return.

        Returns:
            RetrievalResponse with results.
        """
        if index_name:
            return self._retrieve_from_index(query, index_name, top_k)
        else:
            # Search all indexes and combine results
            return self._retrieve_from_all_indexes(query, top_k)

    def _retrieve_from_index(
        self,
        query: str,
        index_name: str,
        top_k: int,
    ) -> RetrievalResponse:
        """
        Retrieve from a specific index.

        Args:
            query: Search query.
            index_name: Index to search.
            top_k: Number of results.

        Returns:
            RetrievalResponse with results.
        """
        index = self.indexer.get_index(index_name)

        if index is None:
            logger.warning(f"Index {index_name} not found")
            return RetrievalResponse(
                query=query,
                index_used=index_name,
                total_found=0,
                retrieval_type="single_index",
            )

        try:
            # Create retriever from index
            retriever = index.as_retriever(similarity_top_k=top_k)

            # Perform retrieval
            nodes = retriever.retrieve(query)

            # Convert to results
            results = []
            for node in nodes:
                result = RetrievalResult(
                    content=node.get_content(),
                    metadata=node.metadata or {},
                    score=node.score if hasattr(node, 'score') and node.score else 0.0,
                    source_index=index_name,
                    node_id=node.node_id if hasattr(node, 'node_id') else "",
                )
                results.append(result)

            return RetrievalResponse(
                results=results,
                query=query,
                index_used=index_name,
                total_found=len(results),
                retrieval_type="single_index",
            )

        except Exception as e:
            logger.error(f"Error retrieving from {index_name}: {e}")
            return RetrievalResponse(
                query=query,
                index_used=index_name,
                total_found=0,
                retrieval_type="single_index",
            )

    def _retrieve_from_all_indexes(
        self,
        query: str,
        top_k: int,
    ) -> RetrievalResponse:
        """
        Retrieve from all indexes and merge the results.

        Args:
            query: Search query.
            top_k: Number of results per index.

        Returns:
            RetrievalResponse with combined results.
        """
        from .indexer import RAGIndexer

        all_results = []

        for index_name in [
            RAGIndexer.COLLECTION_PARSED_DCK_SYSTEMS,
            RAGIndexer.COLLECTION_TMF_COMPONENTS,
            RAGIndexer.COLLECTION_DCK_SYSTEMS,
        ]:
            response = self._retrieve_from_index(query, index_name, top_k)
            all_results.extend(response.results)

        # Sort by score (descending)
        all_results.sort(key=lambda x: x.score, reverse=True)

        # Limit to top_k total
        all_results = all_results[:top_k]

        return RetrievalResponse(
            results=all_results,
            query=query,
            index_used="all",
            total_found=len(all_results),
            retrieval_type="multi_index",
        )

    # === Strategy Retrieval Methods ("copy when possible") ===

    def retrieve_similar_systems(
        self,
        task_description: str,
        top_k: int = 5,
    ) -> RetrievalResponse:
        """
        Retrieve similar system designs based on the "copy when possible" principle.

        Retrieves similar TRNSYS system configurations based on task description.
        Uses LLM to extract 3-6 English keywords from task_description for
        optimal vector similarity matching.

        Args:
            task_description: Description of the system to build.
            top_k: Number of results.

        Returns:
            RetrievalResponse with similar systems.
        """
        from .indexer import RAGIndexer

        # Extract English keywords from task_description for better retrieval
        keywords = self._extract_english_keywords(task_description)

        # Enhance query for system-level search
        enhanced_query = f"TRNSYS system configuration: {keywords}"

        response = self._retrieve_from_index(
            enhanced_query,
            RAGIndexer.COLLECTION_PARSED_DCK_SYSTEMS,
            top_k,
        )
        response.retrieval_type = "similar_systems"

        logger.info(f"Retrieved {response.total_found} similar systems")
        return response

    def retrieve_component_specs(
        self,
        component_description: str,
        type_number: Optional[str] = None,
        top_k: int = 5,
    ) -> RetrievalResponse:
        """
        Retrieve component specifications.

        Retrieves TMF component specifications based on description or type number.

        Args:
            component_description: Description of the component needed.
            type_number: Optional specific Type number (e.g., "15", "56").
            top_k: Number of results.

        Returns:
            RetrievalResponse with component specs.
        """
        from .indexer import RAGIndexer

        # Build query
        if type_number:
            query = f"Type {type_number} component: {component_description}"
        else:
            query = f"TRNSYS component: {component_description}"

        response = self._retrieve_from_index(
            query,
            RAGIndexer.COLLECTION_TMF_COMPONENTS,
            top_k,
        )
        response.retrieval_type = "component_specs"

        logger.info(f"Retrieved {response.total_found} component specs")
        return response

    # === Strategy Retrieval Methods ("copy first, then modify") ===

    def retrieve_format_examples(
        self,
        type_ids: List[str],
        section_type: Optional[str] = None,
        top_k: int = 5,
    ) -> RetrievalResponse:
        """
        Retrieve DCK format examples based on the "copy first, then modify" principle.

        Retrieves DCK format examples containing specified component types.
        Used to find format templates that can be modified.

        Args:
            type_ids: List of Type IDs to search for (e.g., ["15", "56"]).
            section_type: Optional section type (e.g., "UNIT", "EQUATIONS").
            top_k: Number of results.

        Returns:
            RetrievalResponse with format examples.
        """
        from .indexer import RAGIndexer

        # Build query
        type_str = ", ".join([f"Type{t}" for t in type_ids])
        if section_type:
            query = f"DCK format example with {type_str}, {section_type} section"
        else:
            query = f"DCK format example with {type_str}"

        response = self._retrieve_from_index(
            query,
            RAGIndexer.COLLECTION_DCK_SYSTEMS,
            top_k,
        )
        response.retrieval_type = "format_examples"

        logger.info(f"Retrieved {response.total_found} format examples")
        return response

    def retrieve_connection_patterns(
        self,
        source_type: str,
        target_type: str,
        top_k: int = 5,
    ) -> RetrievalResponse:
        """
        Retrieve connection examples.

        Retrieves examples of connections between two component types.

        Args:
            source_type: Source component Type ID.
            target_type: Target component Type ID.
            top_k: Number of results.

        Returns:
            RetrievalResponse with connection examples.
        """
        from .indexer import RAGIndexer

        query = f"Connection from Type{source_type} to Type{target_type}"

        # Search dck_systems index only (raw DCK files)
        response = self._retrieve_from_index(
            query,
            RAGIndexer.COLLECTION_DCK_SYSTEMS,
            top_k,
        )
        response.retrieval_type = "connection_examples"

        logger.info(f"Retrieved {response.total_found} connection examples")
        return response

    def retrieve_equation_patterns(
        self,
        equation_purpose: str,
        related_types: Optional[List[str]] = None,
        top_k: int = 5,
    ) -> RetrievalResponse:
        """
        Retrieve EQUATIONS block examples.

        Retrieves EQUATIONS block examples for specific purposes.

        Args:
            equation_purpose: Purpose of the equation (e.g., "control logic").
            related_types: Optional list of related Type IDs.
            top_k: Number of results.

        Returns:
            RetrievalResponse with equation examples.
        """
        from .indexer import RAGIndexer

        # Translate non-English equation_purpose to English
        equation_purpose_en = self._translate_to_english(equation_purpose)

        # Build query
        if related_types:
            type_str = ", ".join([f"Type{t}" for t in related_types])
            query = f"EQUATIONS block for {equation_purpose_en} with {type_str}"
        else:
            query = f"EQUATIONS block for {equation_purpose_en}"

        # Search dck_systems index only (raw DCK files)
        response = self._retrieve_from_index(
            query,
            RAGIndexer.COLLECTION_DCK_SYSTEMS,
            top_k,
        )
        response.retrieval_type = "equation_examples"

        logger.info(f"Retrieved {response.total_found} equation examples")
        return response

    def retrieve_parameter_examples(
        self,
        type_id: str,
        parameter_context: Optional[str] = None,
        top_k: int = 5,
    ) -> RetrievalResponse:
        """
        Retrieve parameter configuration examples.

        Retrieves parameter configuration examples for a specific component type.

        Args:
            type_id: Component Type ID.
            parameter_context: Optional context for parameter usage.
            top_k: Number of results.

        Returns:
            RetrievalResponse with parameter examples.
        """
        from .indexer import RAGIndexer

        # Translate non-English parameter_context to English
        context_en = self._translate_to_english(parameter_context) if parameter_context else None

        # Build query
        if context_en:
            query = f"Type{type_id} parameters for {context_en}"
        else:
            query = f"Type{type_id} parameter configuration"

        # Search dck_systems index only (raw DCK files)
        response = self._retrieve_from_index(
            query,
            RAGIndexer.COLLECTION_DCK_SYSTEMS,
            top_k,
        )
        response.retrieval_type = "parameter_examples"

        logger.info(f"Retrieved {response.total_found} parameter examples")
        return response

    def retrieve_input_examples(
        self,
        type_id: str,
        input_context: Optional[str] = None,
        top_k: int = 5,
    ) -> RetrievalResponse:
        """
        Retrieve input configuration examples.

        Retrieves input configuration examples for a specific component type,
        including input initial values and input connection patterns.

        Args:
            type_id: Component Type ID.
            input_context: Optional context for input usage.
            top_k: Number of results.

        Returns:
            RetrievalResponse with input examples.
        """
        from .indexer import RAGIndexer

        # Translate non-English input_context to English
        context_en = self._translate_to_english(input_context) if input_context else None

        # Build query
        if context_en:
            query = f"Type{type_id} input configuration for {context_en}"
        else:
            query = f"Type{type_id} input configuration and initial values"

        # Search dck_systems index only (raw DCK files)
        response = self._retrieve_from_index(
            query,
            RAGIndexer.COLLECTION_DCK_SYSTEMS,
            top_k,
        )
        response.retrieval_type = "input_examples"

        logger.info(f"Retrieved {response.total_found} input examples")
        return response

    # === Comprehensive Retrieval Method ===

    def retrieve_for_generation(
        self,
        task_description: str,
        planned_types: List[str],
        top_k: int = 3,
    ) -> Dict[str, RetrievalResponse]:
        """
        Perform comprehensive retrieval for DCK generation.

        Performs multiple retrieval operations for DCK generation:
        1. Similar systems (for reference)
        2. Component specs (for selected types)
        3. Format examples (for templates)

        Args:
            task_description: Description of the system to build.
            planned_types: List of planned Type IDs.
            top_k: Number of results per retrieval type.

        Returns:
            Dictionary mapping retrieval types to responses.
        """
        results = {}

        # 1. Retrieve similar systems
        results["similar_systems"] = self.retrieve_similar_systems(
            task_description, top_k
        )

        # 2. Retrieve component specs for planned types
        component_results = []
        for type_id in planned_types[:5]:  # Limit to avoid too many queries
            response = self.retrieve_component_specs(
                f"Type {type_id} component",
                type_number=type_id,
                top_k=2,
            )
            component_results.extend(response.results)

        results["component_specs"] = RetrievalResponse(
            results=component_results[:top_k * 2],
            query=f"Components: {', '.join(planned_types)}",
            index_used="tmf_components",
            total_found=len(component_results),
            retrieval_type="component_specs",
        )

        # 3. Retrieve format examples
        results["format_examples"] = self.retrieve_format_examples(
            planned_types, top_k=top_k
        )

        logger.info(
            f"Comprehensive retrieval completed: "
            f"{sum(r.total_found for r in results.values())} total results"
        )

        return results

    # === Utility Methods ===

    def format_results_for_prompt(
        self,
        response: RetrievalResponse,
        max_chars: int = 4000,
    ) -> str:
        """
        Format retrieval results for an LLM prompt.

        Args:
            response: Retrieval response to format.
            max_chars: Maximum characters in output.

        Returns:
            Formatted string for LLM prompt.
        """
        if not response.results:
            return f"No results found for: {response.query}"

        parts = [f"Retrieved {response.total_found} results for: {response.query}\n"]
        current_chars = len(parts[0])

        for i, result in enumerate(response.results, 1):
            # Format single result
            result_text = f"\n--- Result {i} (score: {result.score:.3f}) ---\n"
            result_text += result.content[:1000]  # Limit individual result

            if current_chars + len(result_text) > max_chars:
                parts.append("\n... [additional results truncated]")
                break

            parts.append(result_text)
            current_chars += len(result_text)

        return "".join(parts)

    def format_comprehensive_results(
        self,
        results: Dict[str, RetrievalResponse],
        max_chars: int = 8000,
    ) -> str:
        """
        Format comprehensive retrieval results for an LLM prompt.

        Args:
            results: Dictionary of retrieval responses.
            max_chars: Maximum characters in output.

        Returns:
            Formatted string for LLM prompt.
        """
        parts = ["=== RAG Retrieval Results ===\n"]
        current_chars = len(parts[0])
        chars_per_section = max_chars // len(results)

        for retrieval_type, response in results.items():
            section_header = f"\n## {retrieval_type.replace('_', ' ').title()} ##\n"
            section_content = self.format_results_for_prompt(
                response, max_chars=chars_per_section
            )

            section_text = section_header + section_content

            if current_chars + len(section_text) > max_chars:
                parts.append("\n... [additional sections truncated]")
                break

            parts.append(section_text)
            current_chars += len(section_text)

        return "".join(parts)


# === Module-level convenience functions ===

_retriever_instance: Optional[RAGRetriever] = None


def get_rag_retriever() -> RAGRetriever:
    """
    Get the RAG retriever instance.

    Returns a singleton instance of RAGRetriever.

    Returns:
        RAGRetriever instance.
    """
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = RAGRetriever()
    return _retriever_instance


def reset_retriever() -> None:
    """
    Reset the retriever instance.

    Clears the cached retriever instance.
    """
    global _retriever_instance
    _retriever_instance = None
    logger.info("Reset RAG retriever")


# === Convenience retrieval functions ===

def search_similar_systems(task_description: str, limit: int = 5) -> RetrievalResponse:
    """Search for similar system designs."""
    retriever = get_rag_retriever()
    return retriever.retrieve_similar_systems(task_description, limit)


def search_component_specs(
    component_description: str,
    type_number: Optional[str] = None,
    limit: int = 5,
) -> RetrievalResponse:
    """Search for component specifications."""
    retriever = get_rag_retriever()
    return retriever.retrieve_component_specs(component_description, type_number, limit)


def search_format_examples(
    type_ids: List[str],
    section_type: Optional[str] = None,
    limit: int = 5,
) -> RetrievalResponse:
    """Search for DCK format examples."""
    retriever = get_rag_retriever()
    return retriever.retrieve_format_examples(type_ids, section_type, limit)


def search_connection_patterns(
    source_type: str,
    target_type: str,
    limit: int = 5,
) -> RetrievalResponse:
    """Search for connection patterns."""
    retriever = get_rag_retriever()
    return retriever.retrieve_connection_patterns(source_type, target_type, limit)


def search_equation_patterns(
    equation_purpose: str,
    related_types: Optional[List[str]] = None,
    limit: int = 5,
) -> RetrievalResponse:
    """Search for EQUATIONS block patterns."""
    retriever = get_rag_retriever()
    return retriever.retrieve_equation_patterns(equation_purpose, related_types, limit)


def search_parameter_examples(
    type_id: str,
    parameter_context: Optional[str] = None,
    limit: int = 5,
) -> RetrievalResponse:
    """Search for parameter configuration examples."""
    retriever = get_rag_retriever()
    return retriever.retrieve_parameter_examples(type_id, parameter_context, limit)


def search_input_examples(
    type_id: str,
    input_context: Optional[str] = None,
    limit: int = 5,
) -> RetrievalResponse:
    """Search for input configuration examples."""
    retriever = get_rag_retriever()
    return retriever.retrieve_input_examples(type_id, input_context, limit)


def search_for_generation(
    task_description: str,
    planned_types: List[str],
    top_k: int = 3,
) -> Dict[str, RetrievalResponse]:
    """Perform comprehensive retrieval for DCK generation."""
    retriever = get_rag_retriever()
    return retriever.retrieve_for_generation(task_description, planned_types, top_k)
