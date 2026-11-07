"""
Proposed MultiAgent_Memory Method for TRNSYS DCK Generation.

This is the core implementation of the Proposed method:
- Multi-Agent System (7 experts + Supervisor)
- Memory System (Mem0 with 1,097 embeddings)
- NO RAG (RAG is used by B2 and B3 methods)

Key Features:
1. Memory-enhanced multi-agent collaboration
2. Iterative repair with memory-based error solutions
3. Experience recording (success/failure)
4. ESR=100% hard requirement

Design Principles:
- LLM-driven decisions (NO hardcoded rules)
- Memory retrieval for context enhancement
- "copy when possible" from memory
- Record all experiences for future reference
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..core.config import Settings, get_settings
from ..core.state import create_initial_state, DCKGeneratorState
from ..evaluation.metrics import (
    DCKMetricsCalculator,
    EvaluationResult,
    MetricResult,
    get_metrics_calculator,
)
from .base_method import BaseMethod, GenerationResult, MethodRegistry

logger = logging.getLogger(__name__)


@MethodRegistry.register("proposed")
class ProposedMultiAgentMemory(BaseMethod):
    """
    Proposed Method: MultiAgent + Memory.

    This is the main experimental method that must achieve ESR=100%.

    Architecture:
    - 7 Expert Agents (Planner, ComponentSelector, ParameterConfigurator,
      ConnectionGenerator, EquationGenerator, LayoutDesigner, Validator)
    - 1 Supervisor Agent (LLM-driven routing)
    - Mem0 Memory System (1,097 embeddings from training set)
    - NO RAG (controlled variable - RAG is used by B3)

    Key Differences from B3:
    - B3 uses RAG tools, Proposed uses Memory tools
    - Same multi-agent framework (control variable principle)
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        max_iterations: Optional[int] = None,
    ):
        """
        Initialize the Proposed method.

        Args:
            settings: Configuration settings.
            max_iterations: Maximum repair iterations. If None, uses performance_config.
        """
        super().__init__(settings=settings, name="ProposedMultiAgentMemory")

        # Use performance_config if max_iterations not explicitly provides
        if max_iterations is None:
            from ..core.performance_config import get_performance_config
            perf_config = get_performance_config()
            self.max_iterations = perf_config.max_workflow_iterations
        else:
            self.max_iterations = max_iterations

        # Lazy initialization
        self._memory_client = None
        self._supervisor = None
        self._metrics_calculator = None
        self._state_integration = None

    @property
    def memory_client(self):
        """Get or create Memory client (disabled - memory module removed)."""
        if self._memory_client is None:
            logger.warning("Memory client disabled - memory module has been removed")
            self._memory_client = None
        return self._memory_client

    @property
    def supervisor(self):
        """Get or create Supervisor agent."""
        if self._supervisor is None:
            try:
                from ..agents import create_multi_agent_system
                self._supervisor = create_multi_agent_system(
                    method="proposed",
                    max_iterations=self.max_iterations,
                )
                logger.info("Multi-agent system initialized (method=proposed)")
            except Exception as e:
                logger.error(f"Failed to initialize Supervisor: {e}")
                raise
        return self._supervisor

    @property
    def metrics_calculator(self):
        """Get or create metrics calculator."""
        if self._metrics_calculator is None:
            self._metrics_calculator = get_metrics_calculator()
        return self._metrics_calculator

    @property
    def state_integration(self):
        """Get or create state-memory integration (disabled - memory module removed)."""
        if self._state_integration is None:
            logger.warning("State integration disabled - memory module has been removed")
            self._state_integration = None
        return self._state_integration

    async def generate(
        self,
        task_description: str,
        task_id: Optional[str] = None,
    ) -> GenerationResult:
        """
        Generate a DCK file using the Proposed method.

        This is the main entry point for DCK generation.

        Flow:
        1. Initialize state with task description
        2. Populate initial memory context
        3. Run multi-agent workflow
        4. Calculate metrics
        5. Store generation experience
        6. Return result

        Args:
            task_description: Natural language description of the system.
            task_id: Optional task identifier.

        Returns:
            GenerationResult with DCK content and metrics.
        """
        start_time = time.time()
        task_id = task_id or self._create_task_id()

        logger.info(f"Starting Proposed method generation for task: {task_id}")

        try:
            # 1. Initialize state
            state = self._initialize_state(task_id, task_description)

            # 2. Populate initial memory context
            state = self._initialize_memory_context(state)

            # 3. Run multi-agent workflow
            final_state = await self._run_multi_agent_workflow(state)

            # 4. Extract results
            dck_content = final_state.get("dck_content", "")
            validation_result = final_state.get("validation_result", "unknown")
            validation_errors = final_state.get("validation_errors", [])
            iterations = final_state.get("iteration", 0)

            # 5. Calculate metrics (if we have reference DCK)
            # Note: Reference DCK is loaded by the runner, not here
            execution_success = validation_result == "success"

            # 6. Store generation experience
            self._store_generation_experience(
                final_state,
                success=execution_success,
            )

            # 7. Build result
            end_time = time.time()
            duration = end_time - start_time

            result = GenerationResult(
                dck_content=dck_content,
                success=execution_success,
                validation_passed=execution_success,
                validation_errors=[str(e) for e in validation_errors],
                execution_success=execution_success,
                iterations=iterations,
                duration_seconds=duration,
                start_time=datetime.fromtimestamp(start_time),
                end_time=datetime.fromtimestamp(end_time),
                components_selected=final_state.get("components", []),
                connections_generated=final_state.get("connections", []),
                equations_generated=final_state.get("equations", []),
            )

            logger.info(
                f"Generation completed for {task_id}: "
                f"success={execution_success}, iterations={iterations}, "
                f"duration={duration:.2f}s"
            )

            return result

        except Exception as e:
            logger.error(f"Generation failed for {task_id}: {e}")
            return GenerationResult(
                dck_content="",
                success=False,
                validation_passed=False,
                validation_errors=[str(e)],
                execution_success=False,
                error_history=[str(e)],
            )

    def generate_sync(
        self,
        task_description: str,
        task_id: Optional[str] = None,
    ) -> GenerationResult:
        """
        Synchronous version of generate.

        Args:
            task_description: Natural language description.
            task_id: Optional task identifier.

        Returns:
            GenerationResult.
        """
        return asyncio.run(self.generate(task_description, task_id))

    def _initialize_state(
        self,
        task_id: str,
        task_description: str,
    ) -> DCKGeneratorState:
        """
        Initialize the LangGraph state.

        Args:
            task_id: Task identifier.
            task_description: Task description.

        Returns:
            Initialized state.
        """
        state = create_initial_state(
            task_id=task_id,
            task_description=task_description,
            max_iterations=self.max_iterations,
        )

        # Add method identifier
        state["method"] = "proposed"

        return state

    def _initialize_memory_context(
        self,
        state: DCKGeneratorState,
    ) -> DCKGeneratorState:
        """
        Populate state with relevant memories.

        This is the key difference from B3 method:
        - Proposed uses Memory for context
        - B3 uses RAG for context

        Args:
            state: Current state.

        Returns:
            State with populated memories.
        """
        if self.state_integration is None:
            logger.warning("State integration not available, skipping memory context")
            return state

        try:
            # Populate initial memories based on task description
            memory_update = self.state_integration.populate_initial_memories(
                state,
                limit=10,
            )

            # Update state with retrieved memories
            state.update(memory_update)

            memory_count = len(memory_update.get("retrieved_memories", []))
            logger.info(f"Populated {memory_count} initial memories")

        except Exception as e:
            logger.warning(f"Failed to populate memory context: {e}")

        return state

    async def _run_multi_agent_workflow(
        self,
        state: DCKGeneratorState,
    ) -> DCKGeneratorState:
        """
        Run the multi-agent workflow.

        Uses the Supervisor to coordinate 7 expert agents.

        Args:
            state: Initial state.

        Returns:
            Final state after workflow completion.
        """
        try:
            # Get compiled graph
            graph = self.supervisor.get_graph()

            # Configure recursion limit to prevent infinite loops
            # This is critical for the entire workflow, not just individual agents
            # Import from performance_config for centralized management
            from ..core.performance_config import get_performance_config
            perf_config = get_performance_config()

            # Trace filter callbacks to hide large state fields from LangSmith
            from ..core.trace_filter import get_trace_filter_callbacks

            # Enhanced config with LangSmith tracing metadata
            config = {
                "recursion_limit": perf_config.workflow_recursion_limit,
                "callbacks": get_trace_filter_callbacks(),
                # LangSmith tracing enhancement
                "metadata": {
                    "task_id": state.get("task_id", "unknown"),
                    "method": "proposed",
                    "max_iterations": self.max_iterations,
                },
                "run_name": f"DCK-Generation-{state.get('task_id', 'unknown')}",
                "tags": ["proposed-method", "multi-agent", "dck-generation"],
            }

            # Run workflow asynchronously with config
            final_state = await graph.ainvoke(state, config=config)

            return final_state

        except Exception as e:
            logger.error(f"Multi-agent workflow failed: {e}")
            state["status"] = "failed"
            state["failure_reason"] = str(e)
            return state

    def _store_generation_experience(
        self,
        state: DCKGeneratorState,
        success: bool,
    ) -> None:
        """
        Store generation experience to Memory.

        Records both successful and failed generations for future reference.

        Args:
            state: Final state.
            success: Whether generation was successful.
        """
        if self.state_integration is None:
            return

        try:
            if success:
                # Store successful generation
                self.state_integration.store_success(state)
                logger.info("Stored successful generation to memory")
            else:
                # Store failure information for learning
                errors = state.get("validation_errors", [])
                if errors:
                    error_msg = "; ".join(str(e) for e in errors[:3])
                    self.state_integration.store_error_fix(
                        state,
                        error_message=error_msg,
                        fix_description="Generation failed - needs investigation",
                    )
                    logger.info("Stored failure information to memory")

        except Exception as e:
            logger.warning(f"Failed to store generation experience: {e}")

    def get_method_info(self) -> Dict[str, Any]:
        """
        Get information about this method.

        Returns:
            Method configuration and capabilities.
        """
        return {
            "name": self.name,
            "method_type": "proposed",
            "description": "MultiAgent + Memory (Proposed Method)",
            "components": {
                "llm": "DeepSeek",
                "rag": False,  # Proposed does NOT use RAG
                "memory": True,  # Proposed uses Memory
                "multi_agent": True,  # 7 experts + Supervisor
            },
            "configuration": {
                "max_iterations": self.max_iterations,
                "memory_embeddings": 1097,  # From training set
                "expert_agents": 7,
            },
            "esr_target": "100%",  # Hard requirement
        }


# === Factory Functions ===

def create_proposed_method(
    settings: Optional[Settings] = None,
    max_iterations: Optional[int] = None,
) -> ProposedMultiAgentMemory:
    """
    Create a Proposed method instance.

    Args:
        settings: Configuration settings.
        max_iterations: Maximum repair iterations.

    Returns:
        Configured ProposedMultiAgentMemory instance.
    """
    return ProposedMultiAgentMemory(
        settings=settings,
        max_iterations=max_iterations,
    )


def get_proposed_method() -> ProposedMultiAgentMemory:
    """
    Get a default Proposed method instance.

    Returns:
        ProposedMultiAgentMemory instance.
    """
    return create_proposed_method()
