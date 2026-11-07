"""
Expert Agents for TRNSYS DCK Generator.

This module provides specialized expert agents for the multi-agent DCK generation system.
Each expert agent is responsible for a specific aspect of DCK file generation.

Expert Agents:
1. PlannerExpert - Analyzes task and creates system plan
2. ComponentSelectorExpert - Selects TRNSYS components
3. ParameterConfiguratorExpert - Configures component parameters
4. ConnectionGeneratorExpert - Generates component connections
5. EquationGeneratorExpert - Generates EQUATIONS blocks
6. LayoutDesignerExpert - Designs component layout
7. ValidatorExpert - Validates and assembles DCK files

Key Design Principles:
- All experts use LangGraph create_react_agent
- Tool configuration based on method (Proposed vs B3)
- No hardcoded DCK generation logic
- LLM-driven decision making

Usage:
    from trnsys_dck_generator.src.agents.experts import (
        create_expert,
        create_all_experts,
        PlannerExpert,
        ComponentSelectorExpert,
        ValidatorExpert,
    )

    # Create single expert
    planner = create_expert("planner", llm, method="proposed")

    # Create all experts
    experts = create_all_experts(llm, method="proposed")
"""

from .base_expert import (
    BaseExpert,
    ExpertConfig,
    create_expert_llm,
)

from .planner import PlannerExpert
from .component_selector import ComponentSelectorExpert
from .parameter_configurator import ParameterConfiguratorExpert
from .connection_generator import ConnectionGeneratorExpert
from .equation_generator import EquationGeneratorExpert
from .layout_designer import LayoutDesignerExpert
from .validator import ValidatorExpert

__all__ = [
    # Base classes
    "BaseExpert",
    "ExpertConfig",
    "create_expert_llm",
    # Expert agents
    "PlannerExpert",
    "ComponentSelectorExpert",
    "ParameterConfiguratorExpert",
    "ConnectionGeneratorExpert",
    "EquationGeneratorExpert",
    "LayoutDesignerExpert",
    "ValidatorExpert",
    # Factory functions
    "create_expert",
    "create_all_experts",
    "EXPERT_CLASSES",
]

# Expert class registry
EXPERT_CLASSES = {
    "planner": PlannerExpert,
    "component_selector": ComponentSelectorExpert,
    "parameter_configurator": ParameterConfiguratorExpert,
    "connection_generator": ConnectionGeneratorExpert,
    "equation_generator": EquationGeneratorExpert,
    "layout_designer": LayoutDesignerExpert,
    "validator": ValidatorExpert,
}


def create_expert(
    expert_name: str,
    llm=None,
    method: str = "proposed",
    **kwargs,
) -> BaseExpert:
    """
    Create a single expert agent.

    Args:
        expert_name: Name of the expert to create.
            Options: "planner", "component_selector", "parameter_configurator",
                    "connection_generator", "equation_generator", "layout_designer",
                    "validator"
        llm: LLM instance to use. If None, creates one based on expert role.
        method: Method type ("proposed" or "b3_multiagent_rag")
        **kwargs: Additional arguments passed to expert constructor.

    Returns:
        Configured expert agent instance.

    Raises:
        ValueError: If expert_name is not recognized.
    """
    if expert_name not in EXPERT_CLASSES:
        raise ValueError(
            f"Unknown expert: {expert_name}. "
            f"Available: {list(EXPERT_CLASSES.keys())}"
        )

    # Create LLM if not provides
    if llm is None:
        llm = create_expert_llm(expert_name)

    expert_class = EXPERT_CLASSES[expert_name]
    return expert_class(llm=llm, method=method, **kwargs)


def create_all_experts(
    llm=None,
    method: str = "proposed",
    **kwargs,
) -> dict:
    """
    Create all expert agents.

    Args:
        llm: Shared LLM instance. If None, creates individual LLMs for each expert.
        method: Method type ("proposed" or "b3_multiagent_rag")
        **kwargs: Additional arguments passed to expert constructors.

    Returns:
        Dictionary mapping expert names to expert instances.
    """
    experts = {}

    for name in EXPERT_CLASSES:
        # Create individual LLM for each expert if not shared
        expert_llm = llm if llm is not None else create_expert_llm(name)
        experts[name] = create_expert(name, llm=expert_llm, method=method, **kwargs)

    return experts
