"""
Agents Module for TRNSYS DCK Generator.

This module provides the multi-agent system for DCK file generation.
It includes the Supervisor agent and 7 expert agents.

Architecture:
- SupervisorAgent: Central coordinator with LLM-driven routing
- Expert Agents: Specialized agents for different aspects of DCK generation
  - PlannerExpert: Analyzes tasks and creates system plans
  - ComponentSelectorExpert: Selects TRNSYS components
  - ParameterConfiguratorExpert: Configures component parameters
  - ConnectionGeneratorExpert: Generates component connections
  - EquationGeneratorExpert: Generates EQUATIONS blocks
  - LayoutDesignerExpert: Designs component layout
  - ValidatorExpert: Validates and assembles DCK files

Method Support:
- Proposed (MultiAgent_Memory): Uses Memory tools
- B3 (MultiAgent_RAG): Uses RAG tools
- Both methods use the same expert agent framework (control variable principle)

Usage:
    from trnsys_dck_generator.src.agents import (
        create_multi_agent_system,
        create_supervisor,
        SupervisorAgent,
    )

    # Create complete system
    system = create_multi_agent_system(method="proposed")

    # Generate DCK
    result = system.generate_sync("Office HVAC system...")
"""

# === Supervisor Agent ===
from .supervisor import (
    SupervisorAgent,
    create_supervisor,
    create_multi_agent_system,
    EXPERT_NAMES,
)

# === Expert Agents ===
from .experts import (
    # Base class
    BaseExpert,
    ExpertConfig,
    create_expert_llm,
    # Individual experts
    PlannerExpert,
    ComponentSelectorExpert,
    ParameterConfiguratorExpert,
    ConnectionGeneratorExpert,
    EquationGeneratorExpert,
    LayoutDesignerExpert,
    ValidatorExpert,
    # Factory functions
    create_expert,
    create_all_experts,
    EXPERT_CLASSES,
)

# === Tools ===
from .tools import (
    # Tool collection functions
    get_tools_for_agent,
    get_tools_for_expert,
    get_tools_for_method,
    get_all_tools,
    # RAG tools
    get_rag_tools,
    get_rag_tools_for_agent,
    # TMF tools
    get_tmf_tools,
    get_tmf_tools_for_agent,
    # Validation tools
    get_validation_tools,
    get_validation_tools_for_agent,
    # Equation tools
    get_equation_tools,
    get_equation_tools_for_agent,
)

__all__ = [
    # === Supervisor ===
    "SupervisorAgent",
    "create_supervisor",
    "create_multi_agent_system",
    "EXPERT_NAMES",
    # === Expert Base ===
    "BaseExpert",
    "ExpertConfig",
    "create_expert_llm",
    # === Expert Agents ===
    "PlannerExpert",
    "ComponentSelectorExpert",
    "ParameterConfiguratorExpert",
    "ConnectionGeneratorExpert",
    "EquationGeneratorExpert",
    "LayoutDesignerExpert",
    "ValidatorExpert",
    # === Expert Factory ===
    "create_expert",
    "create_all_experts",
    "EXPERT_CLASSES",
    # === Tool Functions ===
    "get_tools_for_agent",
    "get_tools_for_expert",
    "get_tools_for_method",
    "get_all_tools",
    "get_rag_tools",
    "get_rag_tools_for_agent",
    "get_tmf_tools",
    "get_tmf_tools_for_agent",
    "get_validation_tools",
    "get_validation_tools_for_agent",
    "get_equation_tools",
    "get_equation_tools_for_agent",
]


# === Convenience Functions ===

def get_agent_info() -> dict:
    """
    Get information about available agents.

    Returns:
        Dictionary with agent information.
    """
    return {
        "supervisor": {
            "name": "SupervisorAgent",
            "description": "Central coordinator with LLM-driven routing",
            "role": "Coordinates expert agents and manages workflow",
        },
        "experts": {
            name: {
                "class": cls.__name__,
                "description": cls.__doc__.split("\n")[0] if cls.__doc__ else "",
            }
            for name, cls in EXPERT_CLASSES.items()
        },
        "total_experts": len(EXPERT_CLASSES),
        "supported_methods": ["proposed", "b3_multiagent_rag"],
    }


def validate_agent_setup() -> dict:
    """
    Validate that all agents are properly configured.

    Returns:
        Dictionary with validation results.
    """
    results = {
        "supervisor": False,
        "experts": {},
        "tools": {},
        "all_valid": False,
    }

    # Check supervisor
    try:
        supervisor = create_supervisor(method="proposed")
        results["supervisor"] = True
    except Exception as e:
        results["supervisor_error"] = str(e)

    # Check experts
    for name in EXPERT_NAMES:
        try:
            expert = create_expert(name, method="proposed")
            results["experts"][name] = True
        except Exception as e:
            results["experts"][name] = False
            results[f"{name}_error"] = str(e)

    # Check tools
    tool_categories = ["memory", "rag", "tmf", "validation", "equation"]
    for category in tool_categories:
        try:
            tools = get_all_tools().get(category, [])
            results["tools"][category] = len(tools)
        except Exception as e:
            results["tools"][category] = 0
            results[f"{category}_tools_error"] = str(e)

    # Overall validation
    results["all_valid"] = (
        results["supervisor"] and
        all(results["experts"].values()) and
        all(v > 0 for v in results["tools"].values())
    )

    return results
