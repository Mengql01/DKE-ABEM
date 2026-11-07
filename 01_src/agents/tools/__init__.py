"""
Agent Tools for TRNSYS DCK Generator.

Provides @tool decorated functions for LangGraph agents.

Tool Categories:
1. RAG Tools (for B2: RAG and B3: MultiAgent_RAG methods)
   - System-level: rag_search_similar_systems, rag_search_component_specs
   - Format-level: rag_search_dck_format_examples, rag_search_connection_examples, etc.
   - Comprehensive: rag_search_global_for_dck_and_tmf

2. TMF Tools (for all methods - component specification)
   - get_tmf_spec: Get component TMF specification
   - validate_parameter_count: Validate parameter count
   - validate_input_count: Validate input count
   - get_component_io_mapping: Get I/O mapping for connections

3. Validation Tools (for validator agent)
   - run_trnsys_validation: Run TrnEXE64.exe validation
   - parse_error_log: Parse error logs
   - analyze_validation_errors: LLM-driven error analysis

4. Equation Tools (for equation_generator agent)
   - get_equation_syntax_rules: Get EQUATIONS syntax rules
   - validate_equation_syntax: Validate equation syntax

Method-Tool Mapping:
- B1: DirectLLM - No tools
- B2: RAG - RAG tools only
- B3: MultiAgent_RAG - RAG tools + TMF tools + Validation tools + Equation tools
- Proposed: MultiAgent_Memory - TMF tools + Validation tools + Equation tools (Memory removed)

Usage:
    from trnsys_dck_generator.src.agents.tools import (
        # RAG tools (for B2 and B3 methods)
        get_rag_tools,
        rag_search_similar_systems,
        # TMF tools (for all methods)
        get_tmf_tools,
        get_tmf_spec,
        # Validation tools
        get_validation_tools,
        run_trnsys_validation,
        # Equation tools
        get_equation_tools,
        get_equation_syntax_rules,
        # Tool collections
        get_tools_for_agent,
        get_tools_for_expert,
        get_all_tools,
    )
"""

# === RAG Tools (for B2: RAG and B3: MultiAgent_RAG) ===
from .rag_tools import (
    # Individual RAG tools
    rag_search_similar_systems,
    rag_search_component_specs,
    rag_search_dck_format_examples,
    rag_search_connection_examples,
    rag_search_equation_examples,
    rag_search_parameter_examples,
    rag_search_input_examples,
    rag_search_global_for_dck_and_tmf,
    # Tool collection functions
    get_rag_tools,
    get_rag_tools_for_b2_method,
    get_rag_tools_for_b3_method,
    get_rag_tools_for_agent,
    get_rag_tool_descriptions,
    validate_rag_tools,
)

# === TMF Tools (for all methods - component specification) ===
from .tmf_tools import (
    # Individual TMF tools
    get_tmf_spec,
    validate_parameter_count,
    validate_input_count,
    get_component_io_mapping,
    # Tool collection functions
    get_tmf_tools,
    get_tmf_tools_for_agent,
    # Utility functions
    clear_tmf_cache,
    get_tmf_stats,
    search_tmf_by_name,
)

# === Validation Tools (for validator agent) ===
# NOTE: Individual validation @tool functions are commented out in validation_tools.py
# because ValidatorExpert uses DCK module directly via Python code.
# Only utility functions (clear_validation_cache, get_validation_stats) and
# tool collection functions (which now return empty lists) are imported.
from .validation_tools import (
    # Individual validation tools - COMMENTED OUT
    # run_trnsys_validation,
    # parse_error_log,
    # analyze_validation_errors,
    # get_validation_status,
    # Tool collection functions (now return empty lists)
    get_validation_tools,
    get_validation_tools_for_agent,
    # Utility functions
    clear_validation_cache,
    get_validation_stats,
)

# === Equation Tools (for equation_generator agent) ===
from .equation_tools import (
    # Individual equation tools
    get_equation_syntax_rules,
    validate_equation_syntax,
    validate_equations_block,
    # search_equation_examples,  # DEPRECATED: Never used in testing
    # Tool collection functions
    get_equation_tools,
    get_equation_tools_for_agent,
)

# === DCK Snippet Tools (for format comparison and fixing) ===
from .dck_snippet_tools import (
    # Individual DCK snippet tools
    search_dck_snippet,
    get_component_special_fields,
    # compare_dck_snippet,  # DEPRECATED: Never used in testing, removed from active toolset
    rank_components_by_usage,
    # Tool collection functions
    get_dck_snippet_tools,
    get_dck_snippet_tools_for_agent,
    # Utility functions
    get_special_fields_for_type,
    clear_dck_snippet_cache,
)

# === Connection Pattern Tools (for connection generation with port names) ===
from .connection_pattern_tools import (
    # Individual connection pattern tools
    search_connection_patterns,
    get_connection_pattern,
    find_dcks_with_components,
    verify_connection_units,
    # Tool collection functions
    get_connection_pattern_tools,
    get_connection_pattern_tools_for_agent,
    # Utility functions
    clear_connection_pattern_cache,
)

__all__ = [
    # === RAG Tools ===
    "rag_search_similar_systems",
    "rag_search_component_specs",
    "rag_search_dck_format_examples",
    "rag_search_connection_examples",
    "rag_search_equation_examples",
    "rag_search_parameter_examples",
    "rag_search_input_examples",
    "rag_search_global_for_dck_and_tmf",
    # === RAG Tool Collections ===
    "get_rag_tools",
    "get_rag_tools_for_b2_method",
    "get_rag_tools_for_b3_method",
    "get_rag_tools_for_agent",
    "get_rag_tool_descriptions",
    "validate_rag_tools",
    # === TMF Tools ===
    "get_tmf_spec",
    "validate_parameter_count",
    "validate_input_count",
    "get_component_io_mapping",
    # === TMF Tool Collections ===
    "get_tmf_tools",
    "get_tmf_tools_for_agent",
    "clear_tmf_cache",
    "get_tmf_stats",
    "search_tmf_by_name",
    # === Validation Tools - COMMENTED OUT ===
    # NOTE: These 4 @tool functions are commented out because ValidatorExpert
    # uses DCK module directly via Python code, not via @tool functions.
    # "run_trnsys_validation",
    # "parse_error_log",
    # "analyze_validation_errors",
    # "get_validation_status",
    # === Validation Tool Collections ===
    "get_validation_tools",
    "get_validation_tools_for_agent",
    "clear_validation_cache",
    "get_validation_stats",
    # === Equation Tools ===
    "get_equation_syntax_rules",
    "validate_equation_syntax",
    "validate_equations_block",
    # "search_equation_examples",  # DEPRECATED: Never used in testing
    # === Equation Tool Collections ===
    "get_equation_tools",
    "get_equation_tools_for_agent",
    # === DCK Snippet Tools ===
    "search_dck_snippet",
    "get_component_special_fields",
    # "compare_dck_snippet",  # DEPRECATED: Never used in testing, removed from active toolset
    "rank_components_by_usage",
    # === DCK Snippet Tool Collections ===
    "get_dck_snippet_tools",
    "get_dck_snippet_tools_for_agent",
    "get_special_fields_for_type",
    "clear_dck_snippet_cache",
    # === Connection Pattern Tools ===
    "search_connection_patterns",
    "get_connection_pattern",
    "find_dcks_with_components",
    "verify_connection_units",
    # === Connection Pattern Tool Collections ===
    "get_connection_pattern_tools",
    "get_connection_pattern_tools_for_agent",
    "clear_connection_pattern_cache",
    # === Combined Tool Functions ===
    "get_tools_for_agent",
    "get_tools_for_expert",
    "get_all_tools",
    "get_tools_for_method",
]


# === Combined Tool Functions ===

def get_tools_for_agent(agent_role: str, method: str = "proposed") -> list:
    """
    Get tools for a specific agent role.

    Returns appropriate tools based on agent role and method.

    Args:
        agent_role: Role of the agent.
            Options: "planner", "component_selector", "parameter_configurator",
                    "connection_generator", "equation_generator", "validator", "supervisor"
        method: Method being used.
            Options: "proposed" (Memory), "b2_rag" (RAG), "b3_multiagent_rag" (RAG)

    Returns:
        List of @tool decorated functions appropriate for the agent and method.
    """
    if method == "proposed" or method == "multiagent_memory":
        # Proposed method: Memory tools removed, use RAG tools as fallback
        return get_rag_tools_for_agent(agent_role)
    elif method in ["b2_rag", "b3_multiagent_rag", "rag"]:
        # B2 and B3 methods use RAG tools
        return get_rag_tools_for_agent(agent_role)
    else:
        # Default to RAG tools
        return get_rag_tools_for_agent(agent_role)


def get_tools_for_expert(expert_name: str, method: str = "proposed") -> list:
    """
    Get the complete toolset for an expert agent.

    Provides the full tool configuration for an expert agent, including:
    - Memory/RAG tools (selected based on method)
    - TMF tools (component specification validation)
    - Validation tools (for the validator agent)
    - Equation tools (for the equation agent)

    Args:
        expert_name: Expert agent name
            Options: "planner", "component_selector", "parameter_configurator",
                    "connection_generator", "equation_generator", "layout_designer",
                    "validator", "supervisor"
        method: Method type
            Options: "proposed" (Memory), "b3_multiagent_rag" (RAG)

    Returns:
        Complete tool list
    """
    tools = []

    # 1. Add Memory or RAG tools based on method
    if method == "proposed" or method == "multiagent_memory":
        # Memory tools removed, use RAG tools as fallback
        tools.extend(get_rag_tools_for_agent(expert_name))
    elif method in ["b3_multiagent_rag", "rag"]:
        tools.extend(get_rag_tools_for_agent(expert_name))

    # 2. Add TMF tools for relevant experts
    tmf_tools = get_tmf_tools_for_agent(expert_name)
    tools.extend(tmf_tools)

    # 3. Add Validation tools for validator
    validation_tools = get_validation_tools_for_agent(expert_name)
    tools.extend(validation_tools)

    # 4. Add Equation tools for equation_generator
    equation_tools = get_equation_tools_for_agent(expert_name)
    tools.extend(equation_tools)

    # 5. Add DCK Snippet tools for format comparison and fixing
    dck_snippet_tools = get_dck_snippet_tools_for_agent(expert_name)
    tools.extend(dck_snippet_tools)

    # 6. Add Connection Pattern tools for connection generation with port names
    conn_pattern_tools = get_connection_pattern_tools_for_agent(expert_name)
    tools.extend(conn_pattern_tools)

    # Remove duplicates while preserving order
    seen = set()
    unique_tools = []
    for tool in tools:
        tool_name = getattr(tool, 'name', str(tool))
        if tool_name not in seen:
            seen.add(tool_name)
            unique_tools.append(tool)

    return unique_tools


def get_all_tools() -> dict:
    """
    Get all available tools.

    Returns:
        Dictionary with all tool categories.
    """
    return {
        "rag": get_rag_tools(),
        "tmf": get_tmf_tools(),
        "validation": get_validation_tools(),
        "equation": get_equation_tools(),
        "dck_snippet": get_dck_snippet_tools(),
        "connection_pattern": get_connection_pattern_tools(),
    }


def get_tools_for_method(method: str) -> list:
    """
    Get tools for a specific method.

    Args:
        method: Method name.
            Options:
            - "b1_direct_llm": No tools (returns empty list)
            - "b2_rag": RAG tools only
            - "b3_multiagent_rag": RAG tools + TMF + Validation + Equation
            - "proposed" or "multiagent_memory": Memory tools + TMF + Validation + Equation

    Returns:
        List of @tool decorated functions for the method.
    """
    method_lower = method.lower()

    if method_lower == "b1_direct_llm" or method_lower == "direct_llm":
        return []  # B1 uses no tools

    elif method_lower == "b2_rag" or method_lower == "rag":
        return get_rag_tools_for_b2_method()

    elif method_lower == "b3_multiagent_rag" or method_lower == "multiagent_rag":
        # B3: RAG + TMF + Validation + Equation
        tools = get_rag_tools_for_b3_method()
        tools.extend(get_tmf_tools())
        tools.extend(get_validation_tools())
        tools.extend(get_equation_tools())
        return tools

    elif method_lower in ["proposed", "multiagent_memory", "memory"]:
        # Proposed: Memory removed, use RAG + TMF + Validation + Equation as fallback
        tools = get_rag_tools()
        tools.extend(get_tmf_tools())
        tools.extend(get_validation_tools())
        tools.extend(get_equation_tools())
        return tools

    else:
        raise ValueError(
            f"Unknown method: {method}. "
            f"Supported: b1_direct_llm, b2_rag, b3_multiagent_rag, proposed"
        )
