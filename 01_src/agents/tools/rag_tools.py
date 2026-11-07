"""
RAG Tools for LangGraph Agents.

Provides @tool decorated functions for RAG retrieval operations.
These tools are used by:
- B2: RAG method (single LLM + RAG)
- B3: MultiAgent_RAG method (multi-agent + RAG)

Note: Proposed: MultiAgent_Memory method does NOT use these RAG tools.
      It uses Memory tools from memory_tools.py instead.

Tool Categories:
1. System-level RAG tools ("copy when possible" principle)
   - search_similar_systems: Find similar TRNSYS system designs
   - search_component_specs: Find component specifications

2. Format-level RAG tools ("copy first, then modify" principle)
   - search_dck_format_examples: Find DCK format templates
   - search_connection_examples: Find connection examples
   - search_parameter_examples: Find parameter configuration examples
   - search_input_examples: Find input configuration examples

3. Comprehensive retrieval tool
   - search_global_for_dck_and_tmf: Combined retrieval for DCK generation
"""

from __future__ import annotations

import logging
from typing import List, Optional

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


# === System-level RAG Tools ("copy when possible") ===

@tool
def rag_search_similar_systems(task_description: str, limit: int = 5) -> str:
    """
    Search for similar TRNSYS system designs.

    Search for similar TRNSYS system configurations based on the task description.
    Use this tool to find reference systems that can be adapted for the current task.
    Supports the "copy when possible" principle.

    IMPORTANT: The task_description should be a SHORT English description with 3-6
    keywords/phrases, NOT the full original task text. Extract key system features first.

    Args:
        task_description: Short English description of the TRNSYS system (3-6 keywords/phrases).
            Do NOT pass the full original task text. Extract key system features first.
            Example: "single zone building, PID controller, heating cooling, lumped capacitance"
        limit: Maximum number of results to return (default: 5).

    Returns:
        Formatted string with similar system designs found.
        Includes system configurations, component lists, and connection patterns.
    """
    try:
        from ...rag.retriever import get_rag_retriever

        retriever = get_rag_retriever()
        response = retriever.retrieve_similar_systems(task_description, top_k=limit)

        if not response.results:
            return f"No similar system design found for '{task_description}'."

        # Format results
        parts = [f"Found {response.total_found} similar system designs:\n"]

        for i, result in enumerate(response.results, 1):
            parts.append(f"\n--- System {i} (similarity: {result.score:.3f}) ---")
            parts.append(result.content[:1500])  # Limit content length

        return "\n".join(parts)

    except Exception as e:
        logger.error(f"Error in rag_search_similar_systems: {e}")
        return f"Error searching similar systems: {str(e)}"


@tool
def rag_search_component_specs(
    component_description: str,
    type_id: Optional[str] = None,
    limit: int = 5,
) -> str:
    """
    Search for TRNSYS component specifications.

    Search for TRNSYS component specifications (TMF data).
    Use this tool to find component parameters, inputs, outputs, and usage guidelines.

    CRITICAL: type_id MUST be the exact TMF filename without .tmf extension.
    The 765 TMF filenames (XXX.tmf -> XXX) are the ONLY valid component IDs.

    Args:
        component_description: Description of the component needed.
            Example: "chiller", "variable-speed pump", "weather data reader"
        type_id: Optional specific component ID (TMF filename without .tmf).
            Example: "Type927", "Type15-6", "Type56", "Type65d"
            If provided, searches for that specific component type.
        limit: Maximum number of results to return (default: 5).

    Returns:
        Formatted string with component specifications.
        Includes parameters, inputs, outputs, and default values.
    """
    try:
        from ...rag.retriever import get_rag_retriever

        retriever = get_rag_retriever()
        # Extract type_number from TMF-based type_id for RAG retriever compatibility
        type_number = None
        if type_id:
            import re as _re
            num_match = _re.search(r'(\d+)', type_id)
            type_number = num_match.group(1) if num_match else type_id
        response = retriever.retrieve_component_specs(
            component_description,
            type_number=type_number,
            top_k=limit,
        )

        if not response.results:
            query_info = type_id if type_id else component_description
            return f"No component specifications found for '{query_info}'."

        # Format results
        parts = [f"Found {response.total_found} component specifications:\n"]

        for i, result in enumerate(response.results, 1):
            parts.append(f"\n--- Component {i} (similarity: {result.score:.3f}) ---")
            parts.append(result.content[:1200])

        return "\n".join(parts)

    except Exception as e:
        logger.error(f"Error in rag_search_component_specs: {e}")
        return f"Error searching component specifications: {str(e)}"


# === Format-level RAG Tools ("copy first, then modify") ===

@tool
def rag_search_dck_format_examples(
    type_ids: List[str],
    section_type: Optional[str] = None,
    limit: int = 5,
) -> str:
    """
    Search for DCK format examples.

    Search for DCK format examples containing specified component types.
    Use this tool to find format templates that can be modified.
    Supports the "copy first, then modify" principle.

    CRITICAL: type_ids MUST be the exact TMF filename without .tmf extension.
    The 765 TMF filenames (XXX.tmf -> XXX) are the ONLY valid component IDs.

    Args:
        type_ids: List of component IDs (TMF filename without .tmf).
            Example: ["Type15-6", "Type56", "Type927"] for weather reader, building, and chiller.
        section_type: Optional section type to focus on.
            Options: "UNIT", "EQUATIONS", "CONSTANTS", "SIMULATION"
        limit: Maximum number of results to return (default: 5).

    Returns:
        Formatted string with DCK format examples.
        Includes raw DCK content that can be used as templates.
    """
    try:
        from ...rag.retriever import get_rag_retriever

        retriever = get_rag_retriever()
        # Extract type numbers from TMF-based IDs for RAG retriever compatibility
        import re as _re
        type_numbers = []
        for tid in type_ids:
            num_match = _re.search(r'(\d+)', tid)
            type_numbers.append(num_match.group(1) if num_match else tid)
        response = retriever.retrieve_format_examples(
            type_numbers,
            section_type=section_type,
            top_k=limit,
        )

        if not response.results:
            type_str = ", ".join(type_ids)
            return f"No DCK format examples containing {type_str} were found."

        # Format results
        type_str = ", ".join(type_ids)
        parts = [f"Found {response.total_found} DCK format examples containing {type_str}:\n"]

        for i, result in enumerate(response.results, 1):
            filename = result.metadata.get("filename", "unknown")
            parts.append(f"\n--- Example {i}: {filename} (similarity: {result.score:.3f}) ---")
            parts.append(result.content[:2000])  # DCK content can be longer

        return "\n".join(parts)

    except Exception as e:
        logger.error(f"Error in rag_search_dck_format_examples: {e}")
        return f"Error searching DCK format examples: {str(e)}"


@tool
def rag_search_connection_examples(
    source_type: str,
    target_type: str,
    limit: int = 5,
) -> str:
    """
    Search for component connection examples.

    Search for examples of connections between two component types.
    Use this tool to understand how to connect components in TRNSYS.

    CRITICAL: Component IDs MUST be the exact TMF filename without .tmf extension.
    Use "Type927" not "927", "Type15-6" not "15".

    Args:
        source_type: Source component ID (TMF filename without .tmf, e.g., "Type927").
        target_type: Target component ID (TMF filename without .tmf, e.g., "Type110").
        limit: Maximum number of results to return (default: 5).

    Returns:
        Formatted string with connection examples.
        Shows how outputs from source connect to inputs of target.
    """
    try:
        from ...rag.retriever import get_rag_retriever
        import re as _re

        retriever = get_rag_retriever()
        # Extract type numbers from TMF-based IDs for RAG retriever compatibility
        src_match = _re.search(r'(\d+)', source_type)
        tgt_match = _re.search(r'(\d+)', target_type)
        src_num = src_match.group(1) if src_match else source_type
        tgt_num = tgt_match.group(1) if tgt_match else target_type
        response = retriever.retrieve_connection_patterns(
            src_num,
            tgt_num,
            top_k=limit,
        )

        if not response.results:
            return f"No connection examples found from {source_type} to {target_type}."

        # Format results
        parts = [
            f"Found {response.total_found} connection examples for {source_type} -> {target_type}:\n"
        ]

        for i, result in enumerate(response.results, 1):
            parts.append(f"\n--- Connection Example {i} (similarity: {result.score:.3f}) ---")
            parts.append(result.content[:1000])

        return "\n".join(parts)

    except Exception as e:
        logger.error(f"Error in rag_search_connection_examples: {e}")
        return f"Error searching connection examples: {str(e)}"


@tool
def rag_search_equation_examples(
    equation_purpose: str,
    related_types: Optional[List[str]] = None,
    limit: int = 5,
) -> str:
    """
    Search for EQUATIONS block examples.

    Search for EQUATIONS block examples for specific purposes.
    Use this tool to find equation examples for control logic, calculations, etc.

    Args:
        equation_purpose: Purpose of the equation.
            Examples: "control logic", "temperature calculation", "flow distribution", "energy statistics"
        related_types: Optional list of related component IDs (TMF filename without .tmf).
            Example: ["Type927", "Type110"] for chiller and pump related equations.
        limit: Maximum number of results to return (default: 5).

    Returns:
        Formatted string with EQUATIONS block examples.
        Shows equation syntax and usage patterns.
    """
    try:
        from ...rag.retriever import get_rag_retriever

        retriever = get_rag_retriever()
        response = retriever.retrieve_equation_patterns(
            equation_purpose,
            related_types=related_types,
            top_k=limit,
        )

        if not response.results:
            return f"No EQUATIONS block examples related to '{equation_purpose}' were found."

        # Format results
        parts = [f"Found {response.total_found} EQUATIONS block examples related to '{equation_purpose}':\n"]

        for i, result in enumerate(response.results, 1):
            parts.append(f"\n--- Equation Example {i} (similarity: {result.score:.3f}) ---")
            parts.append(result.content[:1200])

        return "\n".join(parts)

    except Exception as e:
        logger.error(f"Error in rag_search_equation_examples: {e}")
        return f"Error searching EQUATIONS block examples: {str(e)}"


@tool
def rag_search_parameter_examples(
    type_id: str,
    parameter_context: Optional[str] = None,
    limit: int = 5,
) -> str:
    """
    Search for parameter configuration examples.

    Search for parameter configuration examples for a specific component type.
    Use this tool to find typical parameter values and configurations.

    CRITICAL: type_id MUST be the exact TMF filename without .tmf extension.
    Use "Type927" not "927", "Type15-6" not "15".

    Args:
        type_id: Component ID (TMF filename without .tmf, e.g., "Type927", "Type56", "Type15-6").
        parameter_context: Optional context for parameter usage.
            Examples: "office building", "residential", "industrial facility"
        limit: Maximum number of results to return (default: 5).

    Returns:
        Formatted string with parameter configuration examples.
        Shows typical parameter values and their meanings.
    """
    try:
        from ...rag.retriever import get_rag_retriever
        import re as _re

        retriever = get_rag_retriever()
        # Extract type number from TMF-based ID for RAG retriever compatibility
        num_match = _re.search(r'(\d+)', type_id)
        type_num = num_match.group(1) if num_match else type_id
        response = retriever.retrieve_parameter_examples(
            type_num,
            parameter_context=parameter_context,
            top_k=limit,
        )

        if not response.results:
            return f"No parameter configuration examples found for {type_id}."

        # Format results
        context_info = f" ({parameter_context})" if parameter_context else ""
        parts = [f"Found {response.total_found} parameter configuration examples for {type_id}{context_info}:\n"]

        for i, result in enumerate(response.results, 1):
            parts.append(f"\n--- Parameter Example {i} (similarity: {result.score:.3f}) ---")
            parts.append(result.content[:1200])

        return "\n".join(parts)

    except Exception as e:
        logger.error(f"Error in rag_search_parameter_examples: {e}")
        return f"Error searching parameter configuration examples: {str(e)}"


@tool
def rag_search_input_examples(
    type_id: str,
    input_context: Optional[str] = None,
    limit: int = 5,
) -> str:
    """
    Search for input configuration examples.

    Search for input configuration examples for a specific component type.
    Use this tool to find typical input values, input initial values,
    and input connection patterns in real DCK files.

    CRITICAL: type_id MUST be the exact TMF filename without .tmf extension.
    Use "Type927" not "927", "Type15-6" not "15".

    Args:
        type_id: Component ID (TMF filename without .tmf, e.g., "Type927", "Type56", "Type15-6").
        input_context: Optional context for input usage.
            Examples: "office building", "residential", "chilled water loop"
        limit: Maximum number of results to return (default: 5).

    Returns:
        Formatted string with input configuration examples.
        Shows typical input values and their connection sources.
    """
    try:
        from ...rag.retriever import get_rag_retriever
        import re as _re

        retriever = get_rag_retriever()
        # Extract type number from TMF-based ID for RAG retriever compatibility
        num_match = _re.search(r'(\d+)', type_id)
        type_num = num_match.group(1) if num_match else type_id
        response = retriever.retrieve_input_examples(
            type_num,
            input_context=input_context,
            top_k=limit,
        )

        if not response.results:
            return f"No input configuration examples found for {type_id}."

        # Format results
        context_info = f" ({input_context})" if input_context else ""
        parts = [f"Found {response.total_found} input configuration examples for {type_id}{context_info}:\n"]

        for i, result in enumerate(response.results, 1):
            parts.append(f"\n--- Input Example {i} (similarity: {result.score:.3f}) ---")
            parts.append(result.content[:1200])

        return "\n".join(parts)

    except Exception as e:
        logger.error(f"Error in rag_search_input_examples: {e}")
        return f"Error searching input configuration examples: {str(e)}"


# === Comprehensive Retrieval Tool ===

@tool
def rag_search_global_for_dck_and_tmf(
    task_description: str,
    planned_types: List[str],
) -> str:
    """
    Comprehensive retrieval for DCK generation.

    Perform comprehensive RAG retrieval for DCK generation.
    This tool combines multiple retrieval types:
    1. Similar systems (for reference architecture)
    2. Component specs (for selected types)
    3. Format examples (for templates)

    Use this tool at the beginning of DCK generation to gather all relevant context.

    CRITICAL: planned_types MUST be the exact TMF filename without .tmf extension.
    Use "Type15-6" not "15", "Type927" not "927".

    IMPORTANT: The task_description should be a SHORT English description with 3-6
    keywords/phrases, NOT the full original task text. Extract key system features first.

    Args:
        task_description: Short English description of the TRNSYS system (3-6 keywords/phrases).
            Do NOT pass the full original task text. Extract key system features first.
            Example: "single zone building, PID controller, heating cooling, lumped capacitance"
        planned_types: List of planned component IDs (TMF filename without .tmf).
            Example: ["Type15-6", "Type56", "Type927", "Type110", "Type511"]

    Returns:
        Comprehensive formatted string with all retrieval results.
        Organized by category: similar systems, component specs, format examples.
    """
    try:
        from ...rag.retriever import get_rag_retriever
        import re as _re

        retriever = get_rag_retriever()
        # Extract type numbers from TMF-based IDs for RAG retriever compatibility
        type_numbers = []
        for tid in planned_types:
            num_match = _re.search(r'(\d+)', tid)
            type_numbers.append(num_match.group(1) if num_match else tid)
        results = retriever.retrieve_for_generation(
            task_description,
            type_numbers,
            top_k=3,
        )

        # Format comprehensive results
        formatted = retriever.format_comprehensive_results(results, max_chars=6000)

        return formatted

    except Exception as e:
        logger.error(f"Error in rag_search_global_for_dck_and_tmf: {e}")
        return f"Error during comprehensive retrieval: {str(e)}"


# === Tool Collection Functions ===

def get_rag_tools() -> list:
    """
    Get all RAG tools (8).

    Returns all RAG tools for use with LangGraph agents.
    Used by B2: RAG and B3: MultiAgent_RAG methods.

    Returns:
        List of @tool decorated functions.
    """
    return [
        rag_search_similar_systems,
        rag_search_component_specs,
        rag_search_dck_format_examples,
        rag_search_connection_examples,
        rag_search_equation_examples,
        rag_search_parameter_examples,
        rag_search_input_examples,
        rag_search_global_for_dck_and_tmf,
    ]


def get_rag_tools_for_b2_method() -> list:
    """
    Get the RAG tools dedicated to the B2: RAG method.

    B2 method uses a single LLM plus RAG tools.
    All RAG tools are available.

    Returns:
        List of @tool decorated functions for the B2 method.
    """
    return get_rag_tools()


def get_rag_tools_for_b3_method() -> list:
    """
    Get the RAG tools dedicated to the B3: MultiAgent_RAG method.

    B3 method uses multi-agent framework + RAG tools.
    All RAG tools are available, distributed across agents.

    Returns:
        List of @tool decorated functions for B3 method.
    """
    return get_rag_tools()


def get_rag_tools_for_agent(agent_role: str) -> list:
    """
    Get the RAG tools for a specific agent role.

    Different agents in the B3 method may use different subsets of RAG tools.

    Args:
        agent_role: Role of the agent.
            Options: "planner", "component_selector", "parameter_configurator",
                    "connection_generator", "equation_generator", "validator"

    Returns:
        List of @tool decorated functions appropriate for the agent role.
    """
    role_tools = {
        "planner": [
            rag_search_similar_systems,
            # rag_search_global_for_dck_and_tmf removed: planner phase has no planned_types yet,
            # component_selector will use it later with planned types
        ],
        "component_selector": [
            rag_search_component_specs,
        ],
        "parameter_configurator": [
            rag_search_component_specs,
            rag_search_parameter_examples,
            rag_search_similar_systems,
        ],
        "connection_generator": [
            rag_search_connection_examples,
            rag_search_dck_format_examples,
            rag_search_input_examples,
        ],
        "equation_generator": [
            rag_search_equation_examples,
            rag_search_dck_format_examples,
        ],
        "layout_designer": [
            rag_search_dck_format_examples,
        ],
        "validator": [
            rag_search_component_specs,
            rag_search_dck_format_examples,
            rag_search_connection_examples,
            rag_search_equation_examples,
            rag_search_parameter_examples,
            rag_search_input_examples,
            rag_search_global_for_dck_and_tmf,
        ],
        "supervisor": get_rag_tools(),  # Supervisor has access to all tools
    }

    return role_tools.get(agent_role, get_rag_tools())


# === Utility Functions ===

def get_rag_tool_descriptions() -> dict:
    """
    Get descriptions of all RAG tools.

    Returns:
        Dictionary mapping tool names to descriptions.
    """
    tools = get_rag_tools()
    return {
        tool.name: tool.description
        for tool in tools
    }


def validate_rag_tools() -> dict:
    """
    Validate whether the RAG tools are available.

    Checks if the RAG system is properly initialized.

    Returns:
        Dictionary with validation results.
    """
    results = {
        "tools_count": len(get_rag_tools()),
        "tools_available": True,
        "rag_system_ready": False,
        "errors": [],
    }

    try:
        from ...rag import get_rag_status
        status = get_rag_status()

        results["embedding_valid"] = status.get("embedding", {}).get("valid", False)
        results["indexes"] = status.get("indexes", {})

        # Check if at least one index exists
        indexes = status.get("indexes", {})
        if isinstance(indexes, dict) and not indexes.get("error"):
            has_index = any(
                idx.get("exists", False)
                for idx in indexes.values()
                if isinstance(idx, dict)
            )
            results["rag_system_ready"] = has_index

    except Exception as e:
        results["errors"].append(str(e))
        results["tools_available"] = False

    return results
