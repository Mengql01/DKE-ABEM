"""
LangGraph Studio Entry Point for TRNSYS DCK Generator.

This module exposes the compiled graph for use with `langgraph dev` command.
It provides the graph instance that LangGraph Studio can interact with.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict

# Add project root to path for langgraph dev compatibility
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Initialize environment before importing other modules
from src.core.config import setup_environment
setup_environment()

from src.agents.supervisor import SupervisorAgent, create_multi_agent_system
from src.core.performance_config import get_performance_config
from src.core.state import DCKGeneratorState, create_initial_state
from src.core.trace_filter import get_trace_filter_callbacks

logger = logging.getLogger(__name__)


def make_graph():
    """
    Factory function to create the DCK Generator graph.

    This function is called by LangGraph Studio to create the graph instance.

    Returns:
        Compiled StateGraph for DCK generation.
    """
    perf_config = get_performance_config()

    # Create the multi-agent system with centralized performance settings
    supervisor = create_multi_agent_system(
        method="proposed",
        max_iterations=perf_config.max_workflow_iterations,
    )

    # Return the compiled graph with trace filter callbacks
    compiled = supervisor.get_graph()
    return compiled.with_config(callbacks=get_trace_filter_callbacks())


# Create the graph instance for LangGraph Studio
# This is the entry point referenced in langgraph.json
graph = make_graph()


# Also expose a function to create initial state for testing
def create_test_state(task_description: str, task_id: str = "studio_test") -> Dict[str, Any]:
    """
    Create an initial state for testing in LangGraph Studio.

    Args:
        task_description: Description of the TRNSYS modeling task.
        task_id: Optional task identifier.

    Returns:
        Initial state dictionary.
    """
    perf_config = get_performance_config()
    return create_initial_state(
        task_id=task_id,
        task_description=task_description,
        max_iterations=perf_config.max_workflow_iterations,
    )
