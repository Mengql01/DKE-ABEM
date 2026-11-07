"""
Performance Configuration for TRNSYS DCK Generator.

This module centralizes all performance-related parameters that affect
DCK generation time. Adjust these values to balance speed vs quality.

IMPORTANT: All time-sensitive parameters should be configured here,
not scattered across individual modules.

NOTE: Memory retrieval limits are NOT included here - they remain in
their original locations to preserve retrieval quality.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class PerformanceConfig:
    """
    Centralized performance configuration.

    Adjust these parameters to optimize DCK generation time.
    """

    # =========================================================================
    # WORKFLOW ITERATION LIMITS
    # Controls how many times the multi-agent workflow can iterate
    # =========================================================================

    # Maximum iterations for the entire workflow (Supervisor level)
    # Higher = more repair attempts, but longer time
    # PREVIOUS: 15 (in proposed_full.py), 8 (in dck_generation_03.py)
    # OPTIMIZED: 5 (sufficient for most cases)
    max_workflow_iterations: int = 8

    # =========================================================================
    # EXPERT AGENT TOOL CALL LIMITS
    # Controls maximum tool calls per expert agent before forcing output
    # Lower values = faster but potentially less accurate
    # =========================================================================

    # Planner: Analyzes task and creates system plan
    # RESTORED to original value (PREVIOUS: 3, was optimized to 2, caused issues)
    planner_max_tool_calls: int = 100

    # Component Selector: Selects TRNSYS components
    # RESTORED to original value (PREVIOUS: 5, was optimized to 3, caused issues)
    component_selector_max_tool_calls: int = 500

    # Parameter Configurator: Configures component parameters
    # RESTORED to original value (PREVIOUS: 10, was optimized to 5, caused issues)
    parameter_configurator_max_tool_calls: int = 800

    # Connection Generator: Generates INPUTS connections
    # RESTORED to original value (PREVIOUS: 15, was optimized to 8, caused issues)
    connection_generator_max_tool_calls: int = 800

    # Equation Generator: Generates EQUATIONS blocks
    # RESTORED to original value (PREVIOUS: 8, was optimized to 4, caused issues)
    equation_generator_max_tool_calls: int = 500

    # Layout Designer: Designs component layout
    # PREVIOUS: not explicitly limited
    # RESTORED to original value (PREVIOUS: 5, was optimized to 2, caused issues)
    layout_designer_max_tool_calls: int = 500

    # Validator: Validates DCK files
    # PREVIOUS: not explicitly limited
    # OPTIMIZED: 5 (validation + error analysis)
    validator_max_tool_calls: int = 500

    # =========================================================================
    # LANGGRAPH RECURSION LIMITS
    # Controls maximum recursion depth for LangGraph agents
    # =========================================================================

    # Recursion limit for most expert agents
    expert_recursion_limit: int = 160

    # Higher recursion limit for parameter_configurator and connection_generator
    # (these agents need many tool calls for get_tmf_spec, get_connection_pattern, etc.)
    heavy_expert_recursion_limit: int = 240

    # Recursion limit for the entire workflow graph
    workflow_recursion_limit: int = 100

    # =========================================================================
    # LLM CONFIGURATION
    # Controls LLM behavior and response limits
    # =========================================================================

    # Maximum tokens for LLM response
    # PREVIOUS: 4096 (config.py), 8192 (optimized)
    # UPDATED: 32768 (aligned with most agents' max_tokens)
    # NOTE: parameter_configurator and connection_generator use 65536 (configured in llm_client.py)
    llm_max_tokens: int = 32768

    # LLM request timeout in seconds
    # PREVIOUS: 120 (config.py)
    # OPTIMIZED: 60 (faster failure detection)
    # NOTE: This is the fallback timeout. Use agent_timeout_map for agent-specific timeouts.
    llm_timeout: int = 120

    # Default timeout for agents not explicitly configured in agent_timeout_map
    default_agent_timeout: int = 240

    # Agent-specific timeout configuration (in seconds)
    # Heavy agents (connection_generator, parameter_configurator) need longer timeout
    agent_timeout_map: Dict[str, int] = field(default_factory=lambda: {
        "connection_generator": 360,
        "parameter_configurator": 360,
    })

    # =========================================================================
    # VALIDATION CONFIGURATION
    # Controls TrnEXE64.exe validation behavior
    # =========================================================================

    # TrnEXE64 execution timeout in seconds.
    # This is the SINGLE SOURCE OF TRUTH for the TRNSYS execution timeout and is
    # applied end-to-end across all execution paths:
    #   - Host mode  : client HTTP request timeout (trnsys_runtime.execute_trnsys_on_host)
    #                  AND the Windows host subprocess timeout (trnsys_execution_service)
    #   - Container  : the safety wait limit in dck/validator.py (Wine/native path)
    # PREVIOUS: 120 (validator.py) / 300 (host execution paths, hardcoded)
    # UPDATED : 6000 (100 minutes) to allow complex/slow simulations to finish.
    trnsys_validation_timeout: int = 6000

    # Maximum validation retries per iteration
    # PREVIOUS: 5 (validator.py)
    # OPTIMIZED: 3
    max_validation_retries: int = 5

    # =========================================================================
    # LAYOUT DESIGNER CONFIGURATION
    # Controls layout designer post-processing behavior
    # =========================================================================

    # Enable connection segment merging in layout designer
    # 0 = disabled (no merging)
    # 1 = enabled (merge overlapping connection segments to reduce visual clutter)
    # When enabled, connections with the same source type to the same target
    # will share common path segments, making the layout cleaner
    enable_connection_segment_merging: int = 1

    # =========================================================================
    # DCK LIMITS STATEMENT CONFIGURATION
    # Controls the three values emitted in the DCK "LIMITS" control card:
    #     LIMITS <iterations_before_warning> <warnings_before_error> <iterations_before_trace>
    # These are the SINGLE SOURCE OF TRUTH for the LIMITS statement and are used
    # as the fallback defaults during DCK assembly (src/dck/assembler.py) and as
    # the initial simulation_settings values (src/core/state.py).
    # =========================================================================

    # # of Iterations Before Warning
    # Comment:
    # This value is used in the LIMITS statement to set limit on the number of
    # iterations that will be performed by TRNSYS during a time step before it is
    # determined that the differential equations and/or algebraic equations are not
    # converging and a WARNING message is printed out.
    # See chapter 2 of the TRNSYS manual (Volume I) for details.
    limits_iterations_before_warning: int = 30

    # # of Warnings Before Error
    # Comment:
    # This value is used in the LIMITS statement to set limit on the number of
    # warnings before the simulation is terminated and an ERROR message is printed.
    # See chapter 2 of the TRNSYS manual (Volume I) for details.
    limits_warnings_before_error: int = 500

    # # of Iterations Before Trace
    # Comment:
    # This value is used in the LIMITS statement to set the limit on the number of
    # iterations during a time step before non-converging components are traced.
    # See chapter 2 of the TRNSYS manual (Volume I) for details.
    limits_iterations_before_trace: int = 50


# =========================================================================
# PREVIOUS CONFIGURATION (BACKUP)
# Uncomment to restore previous settings if needed
# =========================================================================

# @dataclass
# class PreviousPerformanceConfig:
#     """Previous configuration - kept for reference/rollback."""
#
#     # Workflow iterations
#     max_workflow_iterations: int = 15  # or 8 in dck_generation_03.py
#
#     # Expert tool call limits
#     planner_max_tool_calls: int = 3
#     component_selector_max_tool_calls: int = 5
#     parameter_configurator_max_tool_calls: int = 10
#     connection_generator_max_tool_calls: int = 15
#     equation_generator_max_tool_calls: int = 8
#     layout_designer_max_tool_calls: int = 5
#     validator_max_tool_calls: int = 10  # estimated
#
#     # Recursion limits
#     expert_recursion_limit: int = 50
#     workflow_recursion_limit: int = 100
#
#     # LLM configuration
#     llm_max_tokens: int = 4096
#     llm_timeout: int = 120
#
#     # Validation configuration
#     trnsys_validation_timeout: int = 120
#     max_validation_retries: int = 5


# Global instance
_performance_config: Optional[PerformanceConfig] = None


def get_performance_config() -> PerformanceConfig:
    """Get the global performance configuration instance."""
    global _performance_config
    if _performance_config is None:
        _performance_config = PerformanceConfig()
    return _performance_config


def set_performance_config(config: PerformanceConfig) -> None:
    """Set a custom performance configuration."""
    global _performance_config
    _performance_config = config


def reset_performance_config() -> None:
    """Reset to default performance configuration."""
    global _performance_config
    _performance_config = PerformanceConfig()


def get_performance_config_dict() -> Dict:
    """Get performance config as dictionary."""
    config = get_performance_config()
    return {
        "max_workflow_iterations": config.max_workflow_iterations,
        "planner_max_tool_calls": config.planner_max_tool_calls,
        "component_selector_max_tool_calls": config.component_selector_max_tool_calls,
        "parameter_configurator_max_tool_calls": config.parameter_configurator_max_tool_calls,
        "connection_generator_max_tool_calls": config.connection_generator_max_tool_calls,
        "equation_generator_max_tool_calls": config.equation_generator_max_tool_calls,
        "layout_designer_max_tool_calls": config.layout_designer_max_tool_calls,
        "validator_max_tool_calls": config.validator_max_tool_calls,
        "expert_recursion_limit": config.expert_recursion_limit,
        "heavy_expert_recursion_limit": config.heavy_expert_recursion_limit,
        "workflow_recursion_limit": config.workflow_recursion_limit,
        "llm_max_tokens": config.llm_max_tokens,
        "llm_timeout": config.llm_timeout,
        "default_agent_timeout": config.default_agent_timeout,
        "agent_timeout_map": config.agent_timeout_map,
        "trnsys_validation_timeout": config.trnsys_validation_timeout,
        "max_validation_retries": config.max_validation_retries,
        "enable_connection_segment_merging": config.enable_connection_segment_merging,
        "limits_iterations_before_warning": config.limits_iterations_before_warning,
        "limits_warnings_before_error": config.limits_warnings_before_error,
        "limits_iterations_before_trace": config.limits_iterations_before_trace,
    }
