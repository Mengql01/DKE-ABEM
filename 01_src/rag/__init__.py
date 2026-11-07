"""
RAG Module for TRNSYS DCK Generator.

This module provides RAG (Retrieval-Augmented Generation) capabilities
for TRNSYS DCK file generation, supporting:
- "copy when possible" principle: Retrieve similar system designs for reference
- "copy first, then modify" principle: Retrieve format templates to adapt

Components:
- embeddings: Embedding configuration using LlamaIndex DashScopeEmbedding
- indexer: Index builder for three specialized indexes
- retriever: Multi-index retrieval with strategy methods

Used by:
- B2: RAG method (single LLM + RAG)
- B3: MultiAgent_RAG method (multi-agent + RAG)

Note: Proposed: MultiAgent_Memory method does NOT use RAG.

Example usage:
    >>> from src.rag import configure_rag_embeddings, build_all_rag_indexes
    >>> from src.rag import search_similar_systems, search_component_specs
    >>>
    >>> # Initialize RAG system
    >>> configure_rag_embeddings()
    >>> build_all_rag_indexes()
    >>>
    >>> # Perform retrieval
    >>> results = search_similar_systems("Office building HVAC system")
    >>> print(results.total_found)
"""

from __future__ import annotations

# === Embedding Configuration ===
from .embeddings import (
    RAGEmbeddingConfig,
    configure_rag_embeddings,
    get_embed_model,
    get_rag_embedding_config,
    reset_embedding_config,
    validate_embedding_config,
)

# === Index Building ===
from .indexer import (
    RAGIndexer,
    build_all_rag_indexes,
    get_rag_indexer,
    reset_indexer,
)

# === Retrieval ===
from .retriever import (
    RAGRetriever,
    RetrievalResponse,
    RetrievalResult,
    get_rag_retriever,
    reset_retriever,
    # Convenience retrieval functions
    search_component_specs,
    search_connection_patterns,
    search_equation_patterns,
    search_for_generation,
    search_format_examples,
    search_input_examples,
    search_parameter_examples,
    search_similar_systems,
)

__all__ = [
    # Embedding
    "RAGEmbeddingConfig",
    "configure_rag_embeddings",
    "get_embed_model",
    "get_rag_embedding_config",
    "reset_embedding_config",
    "validate_embedding_config",
    # Indexer
    "RAGIndexer",
    "build_all_rag_indexes",
    "get_rag_indexer",
    "reset_indexer",
    # Retriever
    "RAGRetriever",
    "RetrievalResponse",
    "RetrievalResult",
    "get_rag_retriever",
    "reset_retriever",
    # Convenience functions
    "search_similar_systems",
    "search_component_specs",
    "search_format_examples",
    "search_connection_patterns",
    "search_equation_patterns",
    "search_parameter_examples",
    "search_input_examples",
    "search_for_generation",
]


def initialize_rag_system(force_rebuild: bool = False) -> None:
    """
    Initialize the full RAG system.

    Configures embeddings and builds all indexes.

    Args:
        force_rebuild: If True, rebuild indexes even if they exist.

    Example:
        >>> from src.rag import initialize_rag_system
        >>> initialize_rag_system()
    """
    import logging

    logger = logging.getLogger(__name__)

    logger.info("Initializing RAG system...")

    # 1. Configure embeddings
    configure_rag_embeddings()
    logger.info("Configured RAG embeddings")

    # 2. Build indexes
    indexes = build_all_rag_indexes(force_rebuild=force_rebuild)
    logger.info(f"Built {len(indexes)} RAG indexes")

    logger.info("RAG system initialized successfully")


def get_rag_status() -> dict:
    """
    Get the status of the RAG system.

    Returns:
        Dictionary with RAG system status information.
    """
    status = {
        "embedding": validate_embedding_config(),
        "indexes": {},
    }

    try:
        indexer = get_rag_indexer()
        status["indexes"] = indexer.get_index_stats()
    except Exception as e:
        status["indexes"] = {"error": str(e)}

    return status
