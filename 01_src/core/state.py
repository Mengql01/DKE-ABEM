"""
LangGraph State Definitions for TRNSYS DCK Generator.

This module defines the state schema used across all agents in the multi-agent system.
Following LangGraph best practices:
- Use TypedDict for state definition
- Use Annotated[list, operator.add] for append-mode lists
- Keep state raw (store facts, not formatted prompts)
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from langchain_core.messages import AnyMessage
from pydantic import BaseModel, Field
from typing_extensions import Annotated, TypedDict


class AgentRole(str, Enum):
    """Roles for different agents in the system."""

    SUPERVISOR = "supervisor"
    PLANNER = "planner"
    COMPONENT_SELECTOR = "component_selector"
    PARAMETER_CONFIGURATOR = "parameter_configurator"
    CONNECTION_GENERATOR = "connection_generator"
    EQUATION_GENERATOR = "equation_generator"
    LAYOUT_DESIGNER = "layout_designer"
    VALIDATOR = "validator"


class TaskStatus(str, Enum):
    """Status of the DCK generation task."""

    PENDING = "pending"
    PLANNING = "planning"
    SELECTING_COMPONENTS = "selecting_components"
    CONFIGURING_PARAMETERS = "configuring_parameters"
    GENERATING_CONNECTIONS = "generating_connections"
    GENERATING_EQUATIONS = "generating_equations"
    DESIGNING_LAYOUT = "designing_layout"
    VALIDATING = "validating"
    COMPLETED = "completed"
    FAILED = "failed"


class ValidationResult(str, Enum):
    """Result of DCK validation."""

    SUCCESS = "success"
    SYNTAX_ERROR = "syntax_error"
    COMPONENT_ERROR = "component_error"
    CONNECTION_ERROR = "connection_error"
    PARAMETER_ERROR = "parameter_error"
    RUNTIME_ERROR = "runtime_error"


class ComponentInfo(BaseModel):
    """Information about a TRNSYS component (Unit)."""

    unit_number: int = Field(..., description="Unique unit number in the DCK")
    type_number: str = Field(..., description="TRNSYS Type number (e.g., '15', '56')")
    type_variant: str = Field(default="", description="Type variant (e.g., 'a', 'b', '6')")
    unit_name: str = Field(..., description="Descriptive name for the unit")
    model_path: str = Field(default="", description="Path to TMF model file")

    # Parameters, inputs, outputs
    parameters: list[dict[str, Any]] = Field(default_factory=list)
    inputs: list[dict[str, Any]] = Field(default_factory=list)
    initial_values: list[float | str] = Field(default_factory=list)

    # Layout information
    position_x: int = Field(default=100)
    position_y: int = Field(default=100)
    layer: str = Field(default="Main")

    # External files
    external_files: list[dict[str, str]] = Field(default_factory=list)

    def get_full_type_id(self) -> str:
        """Get full type identifier (e.g., 'Type15-6', 'Type56')."""
        if self.type_variant:
            return f"Type{self.type_number}{self.type_variant}"
        return f"Type{self.type_number}"


class ConnectionInfo(BaseModel):
    """Information about a connection between components."""

    source_unit: int = Field(..., description="Source unit number")
    source_output: int = Field(..., description="Source output index (1-based)")
    target_unit: int = Field(..., description="Target unit number")
    target_input: int = Field(..., description="Target input index (1-based)")

    # Optional metadata
    source_name: str = Field(default="", description="Name of source output")
    target_name: str = Field(default="", description="Name of target input")


class EquationInfo(BaseModel):
    """Information about an EQUATIONS block."""

    unit_number: int = Field(..., description="Virtual unit number for the equation block")
    unit_name: str = Field(..., description="Name of the equation block")
    equations: list[Any] = Field(default_factory=list, description="List of equations (str or dict with variable/expression/comment)")

    # Layout
    position_x: int = Field(default=100)
    position_y: int = Field(default=100)
    layer: str = Field(default="Main")


def _get_limits_defaults() -> tuple[int, int, int]:
    """Return the DCK LIMITS statement defaults from performance_config.

    Single source of truth: src/core/performance_config.py. Imported lazily to
    avoid any import-order coupling between state and performance_config.
    Returns (iterations_before_warning, warnings_before_error, iterations_before_trace).
    """
    from .performance_config import get_performance_config
    cfg = get_performance_config()
    return (
        cfg.limits_iterations_before_warning,
        cfg.limits_warnings_before_error,
        cfg.limits_iterations_before_trace,
    )


class SimulationSettings(BaseModel):
    """Global simulation settings."""
    start_time: float = Field(default=0, description="Simulation start time (hours)")
    stop_time: float = Field(default=8760, description="Simulation end time (hours)")
    time_step: float = Field(default=1, description="Simulation time step (hours)")

    # Tolerances
    integration_tolerance: float = Field(default=0.001)
    convergence_tolerance: float = Field(default=0.001)

    # Limits
    # NOTE: defaults come from performance_config (single source of truth for the
    # DCK LIMITS statement). See src/core/performance_config.py.
    max_iterations: int = Field(
        default_factory=lambda: _get_limits_defaults()[0]
    )
    max_warnings: int = Field(
        default_factory=lambda: _get_limits_defaults()[1]
    )
    trace_limit: int = Field(
        default_factory=lambda: _get_limits_defaults()[2]
    )

    # Solver settings
    solver_type: int = Field(default=0)
    min_relaxation: float = Field(default=1.0)
    max_relaxation: float = Field(default=1.0)

    # Other settings
    dfq: int = Field(default=1)
    width: int = Field(default=80)
    nan_check: int = Field(default=0)
    overwrite_check: int = Field(default=0)
    time_report: int = Field(default=0)
    eqsolver: int = Field(default=0)


class ValidationError(BaseModel):
    """Represents a validation error from TrnEXE64.exe."""

    error_type: ValidationResult
    message: str
    line_number: Optional[int] = None
    unit_number: Optional[int] = None
    suggestion: str = ""


class DCKGeneratorState(TypedDict):
    """
    Main state for the DCK Generator multi-agent system.

    Following LangGraph best practices:
    - Use Annotated[list, operator.add] ONLY for lists that must accumulate (messages, error_history)
    - Use plain list for fields that should be REPLACED (components, connections, equations)
    - Keep state raw (store facts, not formatted prompts)
    - Single-purpose fields with clear names

    IMPORTANT: components, connections, equations use plain list (replacement mode)
    to prevent exponential growth during repair iterations.
    """

    # === Task Information ===
    task_id: str  # Unique identifier for this generation task
    task_description: str  # Natural language description of the system to model

    # === Agent Communication ===
    messages: list[AnyMessage]  # Conversation history (changed from operator.add to avoid first-round exposure)
    current_agent: str  # Current active agent
    next_agent: str  # Next agent to activate

    # === Task Status ===
    status: str  # Current status (TaskStatus enum value)
    iteration: int  # Current iteration count
    max_iterations: int  # Maximum iterations allowed

    # === Planning Phase ===
    system_analysis: str  # Analysis of the system requirements
    component_plan: list[dict[str, Any]]  # Planned components

    # === Component Selection Phase ===
    # NOTE: Using plain list (not operator.add) to REPLACE components, not accumulate
    components: list[dict[str, Any]]  # Selected components (ComponentInfo dicts)

    # === Connection Generation Phase ===
    # NOTE: Using plain list (not operator.add) to REPLACE connections, not accumulate
    connections: list[dict[str, Any]]  # Generated connections (ConnectionInfo dicts)

    # === Equation Generation Phase ===
    # NOTE: Using plain list (not operator.add) to REPLACE equations, not accumulate
    equations: list[dict[str, Any]]  # Generated equations (EquationInfo dicts)
    constants: list[dict[str, Any]]  # Global constants

    # === Layout Phase ===
    layout_styles: list[dict[str, Any]]  # LINK_STYLE information
    link_style_lines: list[Any]  # Pre-generated LINK_STYLE items from layout_designer (list of dicts with LINK/CONNECTION_SET)

    # === Simulation Settings ===
    simulation_settings: dict[str, Any]  # SimulationSettings dict
    file_bindings: list[dict[str, Any]]  # External file assignments

    # === DCK Content ===
    dck_content: str  # Generated DCK file content
    dck_sections: dict[str, str]  # Individual DCK sections

    # === Validation Phase ===
    validation_result: str  # ValidationResult enum value
    # NOTE: Using plain list (not operator.add) to REPLACE errors each validation
    validation_errors: list[dict[str, Any]]  # List of errors from current validation
    error_history: list[str]  # History of error messages (changed from operator.add to avoid first-round exposure)

    # === Validation Files (persistent, all versions) ===
    # Each entry: {"version_tag": "V1_20260224_060900", "dck_path": "...", "log_path": "...", "lst_path": "...",
    #              "dck_content": "...", "log_content": "...", "lst_content": "..."}
    validation_file_versions: list[dict[str, Any]]  # All validation file versions (changed from operator.add to avoid first-round exposure)

    # === Memory & RAG ===
    retrieved_memories: list[dict[str, Any]]  # Retrieved memories from Mem0
    retrieved_documents: list[dict[str, Any]]  # Retrieved documents from RAG
    similar_dcks: list[dict[str, Any]]  # Similar DCK examples

    # === Metrics ===
    start_timestamp: str  # Task start time
    end_timestamp: str  # Task end time
    token_usage: dict[str, int]  # Token usage statistics

    # === V2.0: Parallel Execution Support ===
    first_round_complete: bool  # Whether first round (planner→validator) is complete
    pending_experts: list[str]  # Queue of experts to execute (for parallel/sequential repair)
    parallel_execution_done: bool  # Flag indicating parallel experts finished
    error_analysis: dict[str, Any]  # LLM analysis of validation errors

    # === V3.0: Repair Round Context ===
    validator_round: int  # Round boundary counter (incremented at each validator execution)
    repair_mode: bool  # Whether current orchestration is in repair mode
    repair_targets: list[str]  # Supervisor-selected target experts for current repair round
    repair_execution_chain: list[str]  # Actual suffix chain to execute until validator
    last_validator_snapshot: dict[str, Any]  # Snapshot of previous validator outputs
    agent_last_outputs: dict[str, dict[str, Any]]  # Per-agent previous output snapshot
    first_round_agent_raw_outputs: dict[str, dict[str, Any]]  # First-round neutral cache for raw agent outputs
    current_round_repair_notes: list[dict[str, Any]]  # Repair notes accumulated in current chain
    repair_notes_history: list[dict[str, Any]]  # Global repair notes history

    # === V3.1: Expert Output Validation & Atomic Retry ===
    agent_retry_counts: dict[str, int]  # Consecutive retry count per expert
    agent_retry_limit: int  # Maximum retries for a single expert output validation
    agent_execution_snapshot: dict[str, Any]  # Pre-execution snapshot for atomic rollback
    agent_execution_snapshot_agent: str  # Expert name associated with the snapshot
    last_dispatched_expert: str  # Last non-validator expert dispatched by supervisor
    agent_validation_failed: bool  # Whether latest expert output validation failed
    expert_output_status: str  # Expert output status: success message or failure reason

    # === V3.2: Human Expert Intervention ===
    human_expert_error_analysis: dict[str, Any]  # Human expert error analysis (empty dict = use LLM analysis)

    # === V3.3: User Cancellation Support ===
    user_cancelled: bool  # Set to True (via State Edit or API) to gracefully stop the workflow

    # === V3.4: LangGraph Store Integration (Memory Optimization) ===
    # Store large objects externally to reduce checkpoint size
    retrieved_document_ids: list[str]  # Document IDs in Store (replaces full content in retrieved_documents)
    similar_dck_refs: list[dict[str, str]]  # Similar DCK references: {"dck_id": "xxx", "store_key": "xxx"}
    validation_file_refs: list[dict[str, str]]  # Validation file refs: {"version_tag": "V1_xxx", "store_namespace": "...", "store_key": "..."}
    message_history_ref: str  # Store key for archived message history (keeps only recent N messages in state)


def create_initial_state(
    task_id: str,
    task_description: str,
    max_iterations: int = 10,
) -> DCKGeneratorState:
    """
    Create an initial state for a new DCK generation task.

    Args:
        task_id: Unique identifier for the task
        task_description: Natural language description of the system
        max_iterations: Maximum validation/retry iterations

    Returns:
        Initialized DCKGeneratorState
    """
    from datetime import datetime

    # LIMITS statement values come from performance_config (single source of truth)
    from .performance_config import get_performance_config
    perf_config = get_performance_config()

    return DCKGeneratorState(
        # Task info
        task_id=task_id,
        task_description=task_description,

        # Agent communication
        current_agent=AgentRole.SUPERVISOR.value,
        next_agent="",

        # Status
        status=TaskStatus.PENDING.value,
        iteration=0,
        max_iterations=max_iterations,

        # Planning
        system_analysis="",
        component_plan=[],

        # Components & Connections
        components=[],
        connections=[],
        equations=[],
        constants=[],

        # Layout
        layout_styles=[],
        link_style_lines=[],

        # Simulation
        simulation_settings={
            "start_time": 0,
            "stop_time": 8760,
            "time_step": 1,
            "integration_tolerance": 0.001,
            "convergence_tolerance": 0.001,
            "max_iterations": perf_config.limits_iterations_before_warning,
            "max_warnings": perf_config.limits_warnings_before_error,
            "trace_limit": perf_config.limits_iterations_before_trace,
        },
        file_bindings=[],

        # DCK content
        dck_content="",
        dck_sections={},

        # Validation
        validation_result="",
        validation_errors=[],

        # Memory & RAG
        retrieved_memories=[],
        retrieved_documents=[],
        similar_dcks=[],

        # Metrics
        start_timestamp=datetime.now().isoformat(),
        end_timestamp="",
        token_usage={"input": 0, "output": 0, "total": 0},

        # V2.0: Parallel Execution Support
        first_round_complete=False,
        pending_experts=[],
        parallel_execution_done=False,
        error_analysis={},

        # V3.0: Repair Round Context
        validator_round=0,
        repair_targets=[],
        repair_execution_chain=[],
        last_validator_snapshot={},
        agent_last_outputs={},
        current_round_repair_notes=[],
        repair_notes_history=[],

        # V3.1: Expert Output Validation & Atomic Retry
        agent_retry_counts={},
        agent_retry_limit=5,
        agent_execution_snapshot={},
        agent_execution_snapshot_agent="",
        last_dispatched_expert="",
        agent_validation_failed=False,
        expert_output_status="",

        # V3.2: Human Expert Intervention
        human_expert_error_analysis={},

        # V3.3: User Cancellation Support
        user_cancelled=False,

        # V3.4: LangGraph Store Integration (Memory Optimization)
        retrieved_document_ids=[],
        similar_dck_refs=[],
        validation_file_refs=[],
        message_history_ref="",
    )


# === Reducer Functions ===
# These functions define how state fields are updated

def add_component(state: DCKGeneratorState, component: ComponentInfo) -> dict:
    """Add a component to the state."""
    return {"components": [component.model_dump()]}


def add_connection(state: DCKGeneratorState, connection: ConnectionInfo) -> dict:
    """Add a connection to the state."""
    return {"connections": [connection.model_dump()]}


def add_equation(state: DCKGeneratorState, equation: EquationInfo) -> dict:
    """Add an equation block to the state."""
    return {"equations": [equation.model_dump()]}


def add_validation_error(state: DCKGeneratorState, error: ValidationError) -> dict:
    """Add a validation error to the state."""
    return {"validation_errors": [error.model_dump()]}
