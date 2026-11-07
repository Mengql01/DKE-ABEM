"""
Supervisor Agent for TRNSYS DCK Generator.

The Supervisor is the central coordinator of the multi-agent system.
It uses LLM-driven routing to decide which expert agent to call next.

Key Responsibilities:
1. Receive task descriptions and initialize workflow
2. Route tasks to appropriate expert agents using LLM decisions
3. Coordinate iterative repair when validation fails
4. Manage workflow state and transitions

Design Principles:
- LLM-driven routing decisions (NO hardcoded if-else rules)
- Supports both Proposed (Memory) and B3 (RAG) methods
- Uses LangGraph StateGraph for workflow management
- Incremental repair, not restart from scratch

V2.0 Changes (2026-01):
- Modified expert sequence: planner → component_selector → equation_generator →
  parameter_configurator → connection_generator → layout_designer → validator
- After first round, Supervisor ALWAYS uses LLM to decide routing (no auto-routing)
- Supports parallel expert execution for repair (Fan-out/Fan-in pattern)
- Results aggregation before calling validator

V2.1 Fix (2026-01-28):
- Fixed state update issue: pending_experts must be set via node return value,
  not by modifying state dict in routing function (LangGraph doesn't persist those changes)
- Routing decision is now stored in state and processed by supervisor node
"""

from __future__ import annotations

import asyncio
import copy
import concurrent.futures
import json
import logging
import os
import re
from typing import Any, Callable, Dict, List, Literal, Optional, Union

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph

from ..core.trnsys_runtime import is_wine_mode, translate_path_to_wine

logger = logging.getLogger(__name__)

# Expert agent names (excluding validator for repair routing)
EXPERT_NAMES = [
    "planner",
    "component_selector",
    "equation_generator",
    "parameter_configurator",
    "connection_generator",
    "layout_designer",
    "validator",
]

# Experts that can be called for repair (planner is first-round only, validator is always chain tail)
REPAIR_EXPERTS = [
    "component_selector",
    "equation_generator",
    "parameter_configurator",
    "connection_generator",
    "layout_designer",
]

# Fixed post-selection repair flow (from selected earliest expert to validator)
REPAIR_BASE_FLOW = [
    "component_selector",
    "equation_generator",
    "parameter_configurator",
    "connection_generator",
    "layout_designer",
    "validator",
]

# Experts whose outputs must be validated immediately after execution
OUTPUT_VALIDATION_EXPERTS = [
    "planner",
    "component_selector",
    "equation_generator",
    "parameter_configurator",
    "connection_generator",
    "layout_designer",
]

# State fields that must be restored when an expert output fails validation
ATOMIC_ROLLBACK_FIELDS = [
    "system_analysis",
    "component_plan",
    "simulation_settings",
    "components",
    "equations",
    "constants",
    "connections",
    "layout_styles",
    "link_style_lines",
    "status",
    "next_agent",
    "current_agent",
    "pending_experts",
    "agent_last_outputs",
    "current_round_repair_notes",
    "repair_notes_history",
    "repair_targets",
    "repair_execution_chain",
    "last_validator_snapshot",
    "parallel_execution_done",
    "repair_mode",
    "validation_result",
    "validation_errors",
    "error_analysis",
]


class SupervisorAgent:
    """
    Supervisor Agent for coordinating multi-agent DCK generation.

    The Supervisor uses LLM to make routing decisions, not hardcoded rules.
    It supports both Proposed (Memory) and B3 (RAG) methods.

    V2.0: Supports parallel expert execution and forced LLM routing after first round.
    V2.1: Fixed state update mechanism for pending_experts.
    """

    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        experts: Optional[Dict[str, Any]] = None,
        method: str = "proposed",
        max_iterations: Optional[int] = None,
        **kwargs,
    ):
        """
        Initialize the Supervisor.

        Args:
            llm: LLM instance for routing decisions.
            experts: Dictionary of expert agent instances.
            method: Method type ("proposed" or "b3_multiagent_rag").
            max_iterations: Maximum repair iterations.
            **kwargs: Additional configuration.
        """
        self.method = method
        if max_iterations is None:
            from ..core.performance_config import get_performance_config
            self.max_iterations = get_performance_config().max_workflow_iterations
        else:
            self.max_iterations = max_iterations
        self._llm = llm
        self._experts = experts
        self._graph = None
        self._kwargs = kwargs

    @property
    def llm(self) -> BaseChatModel:
        """Get or create the LLM instance."""
        if self._llm is None:
            from .experts.base_expert import create_expert_llm
            self._llm = create_expert_llm("supervisor")
        return self._llm

    @property
    def experts(self) -> Dict[str, Any]:
        """Get or create expert agents."""
        if self._experts is None:
            from .experts import create_all_experts
            self._experts = create_all_experts(method=self.method)
        return self._experts

    @property
    def system_prompt(self) -> str:
        """System prompt for routing decisions."""
        return f"""You are the Supervisor of the TRNSYS DCK generation system. Your responsibility is to coordinate multiple expert agents to complete DCK file generation tasks.

[Available Expert Agents]
1. planner - analyze the task and produce a system plan
2. component_selector - select TRNSYS components
3. equation_generator - generate EQUATIONS blocks (equations are special units and should be generated immediately after component selection)
4. parameter_configurator - configure component parameters
5. connection_generator - generate component connections
6. layout_designer - design the component layout
7. validator - validate the DCK file

[Standard Workflow]
1. planner → analyze the task
2. component_selector → select components
3. equation_generator → generate equations (equations are special units)
4. parameter_configurator → configure parameters
5. connection_generator → generate connections
6. layout_designer → design the layout
7. validator → validate the DCK

[Repair Workflow After Validation Failure - Important]
- Analyze the error type and decide which agents need to be involved in the repair
- You may select multiple agents in parallel for repair, separated by commas
- Do not select planner or validator for repair (planner is first-round only, validator has just reported the error)
- After the repair is completed, validator will be called automatically for re-validation
- The maximum number of iterations is {self.max_iterations}

[Routing Decision Principles]
- Decide the next step based on the current state and the validation errors
- When validation fails, you must analyze the error type and determine an appropriate repair strategy
- You may select multiple agents in parallel to improve efficiency
- Use LLM-based analysis rather than hard-coded rules

Please decide which agent should be called next based on the current state.
Output format: {{"next": "agent name, multiple names separated by commas, or FINISH", "reason": "decision rationale"}}

Example outputs:
- Single agent: {{"next": "parameter_configurator", "reason": "Parameter count mismatch"}}
- Multiple agents: {{"next": "parameter_configurator,connection_generator", "reason": "Both parameter and connection issues are present"}}
- Finished: {{"next": "FINISH", "reason": "Validation passed"}}
"""

    @property
    def repair_routing_prompt(self) -> str:
        """System prompt specifically for repair routing decisions."""
        return f"""You are the Supervisor of the TRNSYS DCK generation system. You now need to decide how to repair a DCK file that failed validation.

[Expert Agents Available for Repair] (excluding planner and validator)
1. component_selector - Select appropriate TRNSYS components from 765-component TMF library based on functional requirements, assign unique UNIT numbers, verify component existence and variants using TMF tools
2. equation_generator - Generate EQUATIONS blocks (special components) for calculations, data post-processing, control logic, signal simulation, and component interface bridging using TRNSYS equation syntax
3. parameter_configurator - Configure PARAMETERS blocks (parameter counts must match TMF specs), EXTERNAL_FILES (weather/building files with unique logical units), and SPECIFIC_PROPERTIES for each component
4. connection_generator - Generate INPUTS blocks for all components and equations, create connections using [unit, output] format, handle unconnected inputs with 0,0, ensure input counts match TMF specifications
5. layout_designer - Design component layout with LEFT-TO-RIGHT orientation (horizontal span > vertical span), assign position coordinates based on data flow topology, generate LINK_STYLE blocks with CONNECTION_SET format

[Error Types and Responsible Experts]
- Component selection errors (wrong component type, missing component) → component_selector
- Equation syntax errors, calculation errors, control logic errors → equation_generator
- Parameter count mismatch, parameter value errors, external file path errors, logical unit conflicts → parameter_configurator
- Input count mismatch, connection errors, invalid [unit,output] references → connection_generator
- Layout errors, unit number conflicts, position coordinate errors, LINK_STYLE format errors → layout_designer
- Multiple error types → select multiple experts for repair

[Important Rules]
1. You must choose the appropriate experts based on the error information
2. You may select multiple experts in parallel, separated by commas
3. Do not select planner or validator (planner is first-round only, validator has just reported the error)
4. After the repair is completed, validator will be called automatically

Please analyze the error information and decide which experts should be called for repair.
Output format: {{"next": "expert name, multiple names separated by commas", "reason": "decision rationale"}}
"""

    def build_graph(self) -> StateGraph:
        """
        Build the LangGraph workflow.

        Returns:
            Compiled StateGraph.
        """
        from ..core.state import DCKGeneratorState

        # Create graph
        workflow = StateGraph(DCKGeneratorState)

        # Add supervisor node
        workflow.add_node("supervisor", self._supervisor_node)

        # Add aggregator node for collecting parallel expert results
        workflow.add_node("aggregator", self._aggregator_node)

        # Add expert nodes
        for name in EXPERT_NAMES:
            if name in self.experts:
                workflow.add_node(name, self.experts[name])

        # Add conditional edges from supervisor
        # Now includes "aggregator" as a possible target for parallel execution
        routing_map = {
            "planner": "planner",
            "component_selector": "component_selector",
            "equation_generator": "equation_generator",
            "parameter_configurator": "parameter_configurator",
            "connection_generator": "connection_generator",
            "layout_designer": "layout_designer",
            "validator": "validator",
            "aggregator": "aggregator",
            "FINISH": END,
        }

        workflow.add_conditional_edges(
            "supervisor",
            self._route_to_expert,
            routing_map
        )

        # All experts return to supervisor, unless user cancelled (→ END directly)
        for name in EXPERT_NAMES:
            if name in self.experts:
                workflow.add_conditional_edges(
                    name,
                    self._route_after_expert,
                    {"supervisor": "supervisor", END: END},
                )

        # Aggregator returns to supervisor (which will then route to validator)
        workflow.add_edge("aggregator", "supervisor")

        # Set entry point
        workflow.set_entry_point("supervisor")

        return workflow.compile()

    def get_graph(self) -> StateGraph:
        """Get or create the compiled graph."""
        if self._graph is None:
            self._graph = self.build_graph()
        return self._graph

    async def _supervisor_node(self, state: Dict[str, Any], config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
        """
        Supervisor node (async wrapper with shield for graceful cancellation).
        """
        # Check cancellation before running any logic
        from ..core.cancellation import is_run_cancelled, check_run_status_via_api
        run_id = ""
        thread_id = ""
        if config:
            run_id = str(config.get("run_id", "") or (config.get("configurable") or {}).get("run_id", "") or "")
            thread_id = str((config.get("configurable") or {}).get("thread_id", "") or "")
        if run_id and (is_run_cancelled(run_id) or check_run_status_via_api(thread_id, run_id)):
            logger.info("Supervisor: run %s cancelled, terminating workflow", run_id)
            return {
                "next_agent": "FINISH",
                "status": "failed",
                "current_agent": "supervisor",
                "expert_output_status": "User manually interrupted the workflow",
            }
        if state.get("user_cancelled", False):
            logger.info("Supervisor: user_cancelled=True detected, terminating workflow")
            return {
                "next_agent": "FINISH",
                "status": "failed",
                "current_agent": "supervisor",
                "expert_output_status": "User manually interrupted the workflow",
            }

        loop = asyncio.get_event_loop()
        sync_fut = concurrent.futures.Future()

        def _run_sync():
            try:
                sync_fut.set_result(self._supervisor_node_sync(state))
            except Exception as exc:
                sync_fut.set_exception(exc)

        loop.run_in_executor(None, _run_sync)
        async_fut = asyncio.wrap_future(sync_fut)
        try:
            result = await asyncio.shield(async_fut)
        except asyncio.CancelledError:
            logger.info("Supervisor: CancelledError absorbed by shield, waiting for sync logic to finish")
            result = sync_fut.result(timeout=600)

        return result

    def _supervisor_node_sync(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Supervisor synchronous logic (extracted for shield wrapping).

        First round behavior MUST remain unchanged.
        Repair rounds (round 2+) follow suffix-chain execution to validator.
        """

        # Check if task is complete
        if state.get("status") == "completed":
            return {"next_agent": "FINISH"}

        # Keep iteration as debug/trace counter (not the business stop condition anymore)
        iteration = state.get("iteration", 0) + 1
        logger.info(f"Supervisor iteration(trace): {iteration}/{self.max_iterations}")

        # Business round counter: validator executions define round boundary
        validator_round = state.get("validator_round", 0)
        validation_result = state.get("validation_result", "")

        # New stop condition: validator success OR total rounds reached
        if validation_result == "success":
            return {
                "iteration": iteration,
                "next_agent": "FINISH",
                "status": state.get("status", "completed") or "completed",
                "current_agent": "supervisor",
            }

        if validator_round >= self.max_iterations:
            logger.warning(f"Max validator rounds ({self.max_iterations}) reached, stopping workflow")
            return {
                "status": "failed",
                "next_agent": "FINISH",
                "expert_output_status": f"Reached max validator rounds {self.max_iterations}",
                "iteration": iteration,
                "current_agent": "supervisor",
            }

        # Post-expert output validation gate.
        # On failure (retry / abort): return immediately.
        # On success: carry the updates forward so the normal routing below
        # can call _prepare_expert_dispatch for the NEXT expert, which sets
        # last_dispatched_expert, takes a snapshot, and consumes pending_experts.
        validation_gate_updates: Dict[str, Any] = {}
        validation_gate_result = self._handle_post_expert_validation(state, iteration)
        if validation_gate_result is not None:
            if validation_gate_result.get("agent_validation_failed") or validation_gate_result.get("status") == "failed":
                # Failure path: return immediately (retry with rollback, or abort)
                return validation_gate_result
            # Success path: stash updates, continue normal routing
            validation_gate_updates = validation_gate_result

        # Helper: merge stashed validation-success updates into a routing result.
        # The routing result (from _prepare_expert_dispatch) takes precedence for
        # overlapping keys like agent_execution_snapshot / last_dispatched_expert,
        # while validation-only keys like expert_output_status are preserved.
        def _merge_gate(routing_result: Dict[str, Any]) -> Dict[str, Any]:
            if not validation_gate_updates:
                return routing_result
            merged = dict(validation_gate_updates)
            merged.update(routing_result)
            return merged

        # Execute pending chain items first (used by repair suffix chain)
        pending_experts = state.get("pending_experts", [])
        if pending_experts:
            next_expert = pending_experts[0]
            remaining = pending_experts[1:]
            logger.info(f"Executing pending expert: {next_expert}, remaining: {remaining}")
            return _merge_gate(self._prepare_expert_dispatch(
                state=state,
                iteration=iteration,
                next_agent=next_expert,
                extra_updates={"pending_experts": remaining},
            ))

        # Legacy parallel signal handling (kept for backward compatibility)
        if state.get("parallel_execution_done"):
            logger.info("Parallel execution done, routing to validator")
            return _merge_gate({
                "iteration": iteration,
                "parallel_execution_done": False,
                "next_agent": "validator",
                "current_agent": "supervisor",
            })

        next_agent_from_expert = state.get("next_agent", "")
        is_repair_mode = validator_round > 0

        # FIRST ROUND: keep original sequential routing unchanged
        if not is_repair_mode:
            if next_agent_from_expert and next_agent_from_expert in EXPERT_NAMES + ["FINISH"]:
                if next_agent_from_expert == "FINISH":
                    return _merge_gate({"iteration": iteration, "next_agent": "FINISH", "current_agent": "supervisor"})
                return _merge_gate(self._prepare_expert_dispatch(
                    state=state,
                    iteration=iteration,
                    next_agent=next_agent_from_expert,
                ))

            # Default to planner for the first entry
            if iteration <= 1:
                return _merge_gate(self._prepare_expert_dispatch(
                    state=state,
                    iteration=iteration,
                    next_agent="planner",
                ))

        # REPAIR MODE: use LLM to select targets, then execute suffix chain to validator
        logger.info("Repair mode: using LLM target selection + suffix chain routing")
        routing_result = self._get_repair_routing_with_pending(state)

        # Promote first-round expert instance cache into repair snapshot field when entering repair rounds
        agent_last_outputs = dict(state.get("agent_last_outputs", {}) or {})
        if not agent_last_outputs:
            # Collect first-round raw outputs from expert instance caches
            first_round_cache = {}
            for expert_name, expert_instance in self.experts.items():
                if hasattr(expert_instance, "get_first_round_output_cache"):
                    cached = expert_instance.get_first_round_output_cache(expert_name)
                    if cached:
                        first_round_cache[expert_name] = cached
            if first_round_cache:
                agent_last_outputs = first_round_cache

        next_agent = routing_result.get("next_agent", "")
        dispatch_result = self._prepare_expert_dispatch(
            state=state,
            iteration=iteration,
            next_agent=next_agent,
            extra_updates=routing_result,
        )
        dispatch_result["first_round_complete"] = True
        dispatch_result["repair_mode"] = True
        dispatch_result["current_agent"] = "supervisor"
        dispatch_result["agent_last_outputs"] = agent_last_outputs

        return _merge_gate(dispatch_result)

    def _get_repair_routing_with_pending(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decide repair targets with LLM, then execute suffix base-flow chain to validator.

        Rules:
        - planner is forbidden in repair rounds
        - selected targets are mapped to the earliest target in REPAIR_BASE_FLOW
        - execute from that earliest target through validator
        """
        state_summary = self._build_state_summary(state)

        error_analysis = state.get("error_analysis", {}) or {}
        error_text = ""
        if error_analysis:
            error_text = (
                f"\n[Error Analysis]\n"
                f"Root cause: {error_analysis.get('root_cause', 'unknown')}\n"
                f"Suggested repair experts: {error_analysis.get('responsible_agents', [])}\n"
                f"Repair suggestions: {error_analysis.get('fix_suggestions', [])}"
            )

        # Include human expert error analysis in routing context if provided
        human_ea = state.get("human_expert_error_analysis", {}) or {}
        if any(human_ea.get(k) for k in ("priority", "root_cause", "fix_suggestions", "responsible_agents")):
            error_text += (
                f"\n\n[HUMAN EXPERT ERROR ANALYSIS (HIGH PRIORITY)]"
                f"\nRoot cause: {human_ea.get('root_cause', '')}"
                f"\nResponsible agents: {human_ea.get('responsible_agents', [])}"
                f"\nFix suggestions: {human_ea.get('fix_suggestions', [])}"
                f"\nPriority: {human_ea.get('priority', '')}"
            )

        prompt = f"""{self.repair_routing_prompt}

[Current State]
{state_summary}
{error_text}

Please select only the experts that need to take responsibility for fixing the problem. Do not select planner or validator.
Output JSON format: {{"next": "expert name, multiple names separated by commas", "reason": "decision rationale"}}
"""

        try:
            from langchain_core.runnables import RunnableConfig

            config = RunnableConfig(
                tags=["supervisor", "repair-routing-decision"],
                metadata={
                    "iteration": state.get("iteration", 0),
                    "validator_round": state.get("validator_round", 0),
                    "current_agent": state.get("current_agent", "none"),
                    "status": state.get("status", "pending"),
                    "validation_result": state.get("validation_result", "unknown"),
                },
                run_name="Supervisor-Repair-Routing",
            )

            response = self.llm.invoke([HumanMessage(content=prompt)], config=config)
            content = response.content if hasattr(response, "content") else str(response)
            decision = self._parse_routing_decision(content)

            selected_targets = self._normalize_repair_targets(
                raw_next=decision.get("next", ""),
                fallback_targets=error_analysis.get("responsible_agents", []),
            )

            if not selected_targets:
                selected_targets = ["connection_generator"]

            repair_chain = self._build_repair_execution_chain(selected_targets)
            if not repair_chain:
                repair_chain = ["connection_generator", "layout_designer", "validator"]

            logger.info(
                "Repair routing selected targets=%s, chain=%s, reason=%s",
                selected_targets,
                repair_chain,
                decision.get("reason", ""),
            )

            return {
                "next_agent": repair_chain[0],
                "pending_experts": repair_chain[1:],
                "repair_targets": selected_targets,
                "repair_execution_chain": repair_chain,
                "current_round_repair_notes": [],
            }

        except Exception as e:
            logger.error(f"LLM repair routing error: {e}")
            fallback_chain = ["connection_generator", "layout_designer", "validator"]
            return {
                "next_agent": fallback_chain[0],
                "pending_experts": fallback_chain[1:],
                "repair_targets": ["connection_generator"],
                "repair_execution_chain": fallback_chain,
                "current_round_repair_notes": [],
            }

    def _normalize_repair_targets(self, raw_next: Any, fallback_targets: Any) -> List[str]:
        """Normalize LLM/fallback targets into valid repair experts."""
        candidates: List[str] = []

        if isinstance(raw_next, str) and raw_next.strip():
            candidates.extend([item.strip() for item in raw_next.split(",") if item.strip()])
        elif isinstance(raw_next, list):
            candidates.extend([str(item).strip() for item in raw_next if str(item).strip()])

        if not candidates and fallback_targets:
            if isinstance(fallback_targets, list):
                candidates.extend([str(item).strip() for item in fallback_targets if str(item).strip()])
            elif isinstance(fallback_targets, str):
                candidates.extend([item.strip() for item in fallback_targets.split(",") if item.strip()])

        normalized: List[str] = []
        for candidate in candidates:
            if candidate in REPAIR_EXPERTS and candidate not in normalized:
                normalized.append(candidate)

        return normalized

    def _build_repair_execution_chain(self, selected_targets: List[str]) -> List[str]:
        """Build suffix execution chain from earliest selected target to validator."""
        if not selected_targets:
            return []

        indices = [REPAIR_BASE_FLOW.index(agent) for agent in selected_targets if agent in REPAIR_BASE_FLOW]
        if not indices:
            return []

        start_idx = min(indices)
        return REPAIR_BASE_FLOW[start_idx:]

    def _aggregator_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Aggregator node that collects results from parallel expert execution.

        This node is called after all parallel experts have completed.
        It marks parallel execution as done so supervisor routes to validator.
        """
        logger.info("Aggregator: Collecting parallel expert results, will route to validator next")

        # Mark that parallel execution is complete
        # The supervisor will then route to validator
        return {
            "parallel_execution_done": True,
            "current_agent": "aggregator",
        }

    def _handle_post_expert_validation(self, state: Dict[str, Any], iteration: int) -> Optional[Dict[str, Any]]:
        """Validate the output of the latest executed non-validator expert before continuing."""
        executed_agent = state.get("last_dispatched_expert", "")
        if executed_agent not in OUTPUT_VALIDATION_EXPERTS:
            return None

        current_agent = state.get("current_agent", "")
        if current_agent != executed_agent:
            return None

        if self._is_agent_output_valid(state, executed_agent):
            logger.info("Output validation passed for expert: %s", executed_agent)
            retry_counts = dict(state.get("agent_retry_counts", {}) or {})
            if executed_agent in retry_counts:
                retry_counts.pop(executed_agent, None)
            return {
                "iteration": iteration,
                "current_agent": "supervisor",
                "agent_retry_counts": retry_counts,
                "agent_validation_failed": False,
                "agent_execution_snapshot": {},
                "agent_execution_snapshot_agent": "",
                "expert_output_status": f"{executed_agent} output successfully",
            }

        retry_counts = dict(state.get("agent_retry_counts", {}) or {})
        retry_count = retry_counts.get(executed_agent, 0) + 1
        retry_counts[executed_agent] = retry_count
        retry_limit = state.get("agent_retry_limit", 5) or 5
        logger.warning(
            "Output validation failed for expert %s (%s/%s)",
            executed_agent,
            retry_count,
            retry_limit,
        )

        if retry_count >= retry_limit:
            return {
                "iteration": iteration,
                "status": "failed",
                "next_agent": "FINISH",
                "current_agent": "supervisor",
                "agent_retry_counts": retry_counts,
                "agent_validation_failed": True,
                "expert_output_status": self._get_agent_output_validation_error(executed_agent, retry_count, retry_limit),
            }

        return self._rollback_and_retry_expert(
            state=state,
            expert_name=executed_agent,
            iteration=iteration,
            retry_count=retry_count,
            retry_counts=retry_counts,
        )

    def _prepare_expert_dispatch(
        self,
        state: Dict[str, Any],
        iteration: int,
        next_agent: str,
        extra_updates: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Prepare supervisor state updates before dispatching the next node."""
        updates = {
            "iteration": iteration,
            "next_agent": next_agent,
            "current_agent": "supervisor",
        }
        if extra_updates:
            updates.update(extra_updates)

        if next_agent in OUTPUT_VALIDATION_EXPERTS:
            updates.update(self._build_expert_execution_guard_updates(state, next_agent))
        else:
            updates["last_dispatched_expert"] = ""
            updates["agent_execution_snapshot"] = {}
            updates["agent_execution_snapshot_agent"] = ""
            updates["agent_validation_failed"] = False

        return updates

    def _build_expert_execution_guard_updates(self, state: Dict[str, Any], expert_name: str) -> Dict[str, Any]:
        """Save a deep snapshot before dispatching an expert so failures can be rolled back atomically."""
        snapshot = {
            field: copy.deepcopy(state[field])
            for field in ATOMIC_ROLLBACK_FIELDS
            if field in state
        }
        return {
            "agent_execution_snapshot": snapshot,
            "agent_execution_snapshot_agent": expert_name,
            "last_dispatched_expert": expert_name,
            "agent_validation_failed": False,
        }

    def _rollback_and_retry_expert(
        self,
        state: Dict[str, Any],
        expert_name: str,
        iteration: int,
        retry_count: int,
        retry_counts: Dict[str, int],
    ) -> Dict[str, Any]:
        """Restore the pre-execution snapshot and re-dispatch the same expert."""
        snapshot_agent = state.get("agent_execution_snapshot_agent", "")
        snapshot = state.get("agent_execution_snapshot", {}) or {}

        if snapshot_agent != expert_name or not snapshot:
            logger.warning(
                "Missing or mismatched execution snapshot for %s, retrying without rollback snapshot",
                expert_name,
            )
            return {
                "iteration": iteration,
                "next_agent": expert_name,
                "current_agent": "supervisor",
                "last_dispatched_expert": expert_name,
                "agent_retry_counts": retry_counts,
                "agent_validation_failed": True,
                "expert_output_status": self._get_agent_output_validation_error(
                    expert_name,
                    retry_count,
                    state.get("agent_retry_limit", 5) or 5,
                ),
            }

        rolled_back_state = {
            field: copy.deepcopy(snapshot.get(field))
            for field in ATOMIC_ROLLBACK_FIELDS
            if field in snapshot
        }
        rolled_back_state.update(
            {
                "iteration": iteration,
                "next_agent": expert_name,
                "current_agent": "supervisor",
                "last_dispatched_expert": expert_name,
                "agent_retry_counts": retry_counts,
                "agent_validation_failed": True,
                "expert_output_status": self._get_agent_output_validation_error(
                    expert_name,
                    retry_count,
                    state.get("agent_retry_limit", 5) or 5,
                ),
            }
        )

        logger.info("Rolled back state and retrying expert: %s", expert_name)
        return rolled_back_state

    def _is_agent_output_valid(self, state: Dict[str, Any], agent_name: str) -> bool:
        """Validate that an expert produced its required non-empty state output."""
        if agent_name == "planner":
            return bool((state.get("system_analysis") or "").strip())

        if agent_name == "component_selector":
            components = state.get("components")
            if not isinstance(components, list) or len(components) == 0:
                return False
            # Additionally verify that ALL model_paths point to real .tmf files
            return self._validate_component_model_paths(components)

        if agent_name == "equation_generator":
            equations = state.get("equations")
            return isinstance(equations, list) and len(equations) > 0

        if agent_name == "parameter_configurator":
            components = state.get("components")
            if not isinstance(components, list):
                return False
            return any(isinstance(comp.get("parameters"), list) and len(comp.get("parameters", [])) > 0 for comp in components if isinstance(comp, dict))

        if agent_name == "connection_generator":
            components = state.get("components")
            if not isinstance(components, list):
                return False
            has_inputs = any(
                isinstance(comp.get("inputs"), list) and len(comp.get("inputs", [])) > 0
                for comp in components if isinstance(comp, dict)
            )
            connections = state.get("connections")
            has_connections = isinstance(connections, list) and len(connections) > 0
            return has_inputs and has_connections

        if agent_name == "layout_designer":
            layout_styles = state.get("layout_styles")
            return isinstance(layout_styles, list) and len(layout_styles) > 0

        return True

    def _validate_component_model_paths(self, components: List[Dict[str, Any]]) -> bool:
        """Check that every component has a model_path pointing to an existing .tmf file.

        Returns True only when ALL components have a valid, resolvable model_path.
        """
        for comp in components:
            if not isinstance(comp, dict):
                logger.warning("Component is not a dict, model_path validation failed")
                return False
            model_path = comp.get("model_path", "")
            if not model_path or not isinstance(model_path, str):
                logger.warning(
                    "Component %s has empty or missing model_path",
                    comp.get("type_number", "unknown"),
                )
                return False
            resolved = self._resolve_model_path(model_path)

            # Check if resolved path exists
            # Note: _resolve_model_path already returns the correct format for the current environment
            # (Linux path in Wine mode, Windows path in native Windows mode)
            if not os.path.isfile(resolved):
                if is_wine_mode():
                    logger.warning(
                        "Component %s model_path does not exist: %s (resolved: %s)",
                        comp.get("type_number", "unknown"),
                        model_path,
                        resolved,
                    )
                else:
                    logger.warning(
                        "Component %s model_path does not exist: %s (resolved: %s)",
                        comp.get("type_number", "unknown"),
                        model_path,
                        resolved,
                    )
                return False
        return True

    def _resolve_model_path(self, model_path: str) -> str:
        """Resolve a model_path (relative or absolute) to a filesystem path.

        Relative paths produced by tmf_tools use the format:
            .\\Weather\\Type15-6.tmf   (relative to the Proformas directory)
        Absolute paths may look like:
            d:\\TRNSYS18_04\\Studio\\Proformas\\Weather\\Type15-6.tmf

        The Proformas base directory is derived from the TRNSYS installation
        configured in settings (trnsys_exe_path -> ../../Studio/Proformas).
        """
        # Normalize path separators
        normalized = model_path.replace("/", os.sep).replace("\\\\", os.sep).replace("\\", os.sep)

        # If it is already an absolute path, return directly
        if os.path.isabs(normalized):
            return normalized

        # Relative path: strip leading ".\" or "./" prefix
        if normalized.startswith("." + os.sep):
            normalized = normalized[2:]

        # Build Proformas base directory from config
        proformas_dir = self._get_proformas_dir()
        return os.path.join(proformas_dir, normalized)

    def _get_proformas_dir(self) -> str:
        """Return the TRNSYS Proformas directory path.

        Derived from config trnsys_exe_path:
            D:\\TRNSYS18_04\\Exe\\TrnEXE64.exe -> D:\\TRNSYS18_04\\Studio\\Proformas
        Falls back to a sensible default if config is unavailable.

        In Wine mode, translates Windows path to Linux mount path.
        """
        if not hasattr(self, "_proformas_dir_cache"):
            try:
                from ..core.config import get_settings
                settings = get_settings()
                trnsys_root = settings.trnsys_exe_path.parent.parent
                proformas_path = trnsys_root / "Studio" / "Proformas"
                proformas_str = str(proformas_path)

                # Wine mode: translate to Linux path ONLY if it's a Windows path
                if is_wine_mode():
                    # Check if path is already in Linux format (starts with /)
                    if proformas_str.startswith("/"):
                        # Already a Linux path, use directly
                        self._proformas_dir_cache = proformas_str
                    else:
                        # Windows path, needs translation
                        self._proformas_dir_cache = translate_path_to_wine(proformas_str)
                else:
                    self._proformas_dir_cache = proformas_str
            except Exception:
                # Fallback: use a common default
                if is_wine_mode():
                    self._proformas_dir_cache = "/mnt/trnsys/TRNSYS18_04/Studio/Proformas"
                else:
                    self._proformas_dir_cache = r"D:\TRNSYS18_04\Studio\Proformas"
                logger.warning(
                    "Could not resolve Proformas dir from config, using default: %s",
                    self._proformas_dir_cache,
                )
        return self._proformas_dir_cache

    def _get_agent_output_validation_error(self, agent_name: str, retry_count: int, retry_limit: int) -> str:
        """Build a clear failure reason for empty or missing structured output."""
        field_desc = {
            "planner": "system_analysis is empty or missing",
            "component_selector": "components is empty/missing or one or more model_path values point to non-existent .tmf files",
            "equation_generator": "equations is empty or missing",
            "parameter_configurator": "components.parameters is empty or missing",
            "connection_generator": "components.inputs and/or connections is empty or missing",
            "layout_designer": "layout_styles is empty or missing",
        }.get(agent_name, "required state output is empty or missing")
        return (
            f"Expert {agent_name} produced invalid structured output: {field_desc}. "
            f"Retry {retry_count}/{retry_limit}."
        )

    def _route_to_expert(self, state: Dict[str, Any]) -> str:
        """
        Simple routing function that reads next_agent from state.

        V2.1 Fix: This function now ONLY reads the routing decision from state.
        All state updates (including pending_experts) are done in _supervisor_node.
        """
        next_agent = state.get("next_agent", "planner")

        # Validate the routing decision
        valid_targets = EXPERT_NAMES + ["FINISH", "aggregator"]
        if next_agent in valid_targets:
            return next_agent

        logger.warning(f"Invalid next_agent in state: {next_agent}, defaulting to planner")
        return "planner"

    @staticmethod
    def _route_after_expert(state: Dict[str, Any]) -> str:
        """Route after expert execution: go to END if user cancelled, else supervisor."""
        if state.get("user_cancelled", False):
            return END
        return "supervisor"

    def _build_state_summary(self, state: Dict[str, Any]) -> str:
        """Build a summary of current state for LLM."""
        parts = []

        # Task info
        task_id = state.get("task_id", "unknown")
        task_desc = state.get("task_description", "")[:200]
        parts.append(f"Task ID: {task_id}")
        parts.append(f"Task description: {task_desc}...")

        # Status
        status = state.get("status", "pending")
        iteration = state.get("iteration", 0)
        parts.append(f"Status: {status}")
        parts.append(f"Iteration count: {iteration}")

        # Progress
        components = state.get("components", [])
        parts.append(f"Selected components: {len(components)}")

        # Validation
        validation_result = state.get("validation_result", "pending")
        parts.append(f"Validation result: {validation_result}")

        # Errors
        errors = state.get("validation_errors", [])
        if errors:
            parts.append(f"Error count: {len(errors)}")
            for err in errors[:5]:  # Show more errors for better analysis
                parts.append(f"  - {err.get('message', '')[:150]}")

        # Current/Previous agent
        current = state.get("current_agent", "none")
        parts.append(f"Current agent: {current}")

        # First round status
        first_round = state.get("first_round_complete", False)
        parts.append(f"First round complete: {first_round}")

        # Validator round status
        validator_round = state.get("validator_round", 0)
        parts.append(f"Validator round (validator boundary): {validator_round}/{self.max_iterations}")

        # Repair chain context
        repair_targets = state.get("repair_targets", [])
        if repair_targets:
            parts.append(f"Repair target agents: {repair_targets}")

        repair_chain = state.get("repair_execution_chain", [])
        if repair_chain:
            parts.append(f"Current repair execution chain: {repair_chain}")

        return "\n".join(parts)

    def _parse_routing_decision(self, content: str) -> Dict[str, Any]:
        """Parse routing decision from LLM response."""
        try:
            # Try to extract JSON
            json_match = re.search(r'\{[^}]+\}', content)
            if json_match:
                return json.loads(json_match.group())

            # Try to parse entire content as JSON
            return json.loads(content)

        except json.JSONDecodeError:
            # Extract next agent from text
            for name in EXPERT_NAMES + ["FINISH"]:
                if name.lower() in content.lower():
                    return {"next": name, "reason": "Extracted from text"}

            return {"next": "parameter_configurator", "reason": "Default fallback"}

    def _get_fallback_routing(self, state: Dict[str, Any]) -> str:
        """
        Fallback routing when LLM fails.

        Uses a simple sequential workflow.
        V2.0: Updated sequence with equation_generator after component_selector.
        """
        status = state.get("status", "pending")
        current = state.get("current_agent", "")

        # Sequential workflow (V2.0 order)
        workflow_order = [
            "planner",
            "component_selector",
            "equation_generator",
            "parameter_configurator",
            "connection_generator",
            "layout_designer",
            "validator",
        ]

        # Find next in sequence
        if current in workflow_order:
            idx = workflow_order.index(current)
            if idx < len(workflow_order) - 1:
                return workflow_order[idx + 1]
            else:
                return "FINISH"

        # Default to planner
        return "planner"

    async def generate(self, task_description: str, task_id: str = "") -> Dict[str, Any]:
        """
        Generate a DCK file for the given task.

        Args:
            task_description: Description of the TRNSYS modeling task.
            task_id: Optional task identifier.

        Returns:
            Generation result including DCK content and validation status.
        """
        from ..core.state import create_initial_state

        # Create initial state
        state = create_initial_state(
            task_id=task_id or f"task_{id(task_description)}",
            task_description=task_description,
            max_iterations=self.max_iterations,
        )

        # Get compiled graph
        graph = self.get_graph()

        # Run workflow
        try:
            final_state = await graph.ainvoke(state)
            return self._format_result(final_state)

        except Exception as e:
            logger.error(f"Generation error: {e}")
            return {
                "success": False,
                "error": str(e),
                "dck_content": "",
            }

    def generate_sync(self, task_description: str, task_id: str = "") -> Dict[str, Any]:
        """
        Synchronous version of generate.

        Args:
            task_description: Description of the TRNSYS modeling task.
            task_id: Optional task identifier.

        Returns:
            Generation result.
        """
        from ..core.state import create_initial_state

        # Create initial state
        state = create_initial_state(
            task_id=task_id or f"task_{id(task_description)}",
            task_description=task_description,
            max_iterations=self.max_iterations,
        )

        # Get compiled graph
        graph = self.get_graph()

        # Run workflow
        try:
            final_state = graph.invoke(state)
            return self._format_result(final_state)

        except Exception as e:
            logger.error(f"Generation error: {e}")
            return {
                "success": False,
                "error": str(e),
                "dck_content": "",
            }

    def _format_result(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Format the final result."""
        return {
            "success": state.get("validation_result") == "success",
            "dck_content": state.get("dck_content", ""),
            "task_id": state.get("task_id", ""),
            "status": state.get("status", "unknown"),
            "iterations": state.get("iteration", 0),
            "components": state.get("components", []),
            "validation_result": state.get("validation_result", "unknown"),
            "validation_errors": state.get("validation_errors", []),
        }

    def __repr__(self) -> str:
        return f"SupervisorAgent(method={self.method}, max_iterations={self.max_iterations})"


def create_supervisor(
    method: str = "proposed",
    llm: Optional[BaseChatModel] = None,
    max_iterations: Optional[int] = None,
    **kwargs,
) -> SupervisorAgent:
    """
    Factory function to create a Supervisor agent.

    Args:
        method: Method type ("proposed" or "b3_multiagent_rag").
        llm: Optional LLM instance.
        max_iterations: Maximum repair iterations.
        **kwargs: Additional configuration.

    Returns:
        Configured SupervisorAgent instance.
    """
    return SupervisorAgent(
        llm=llm,
        method=method,
        max_iterations=max_iterations,
        **kwargs,
    )


def create_multi_agent_system(
    method: str = "proposed",
    llm_config: Optional[Dict[str, Any]] = None,
    memory_client: Optional[Any] = None,
    rag_retriever: Optional[Any] = None,
    max_iterations: Optional[int] = None,
) -> SupervisorAgent:
    """
    Create a complete multi-agent system.

    This is the main entry point for creating the DCK generation system.

    Args:
        method: Method type.
            - "proposed": MultiAgent + Memory
            - "b3_multiagent_rag": MultiAgent + RAG
        llm_config: Optional LLM configuration.
        memory_client: Memory client for Proposed method.
        rag_retriever: RAG retriever for B3 method.
        max_iterations: Maximum repair iterations.

    Returns:
        Configured SupervisorAgent with all experts.
    """
    from .experts import create_all_experts
    from .experts.base_expert import create_expert_llm

    # Create supervisor LLM
    supervisor_llm = create_expert_llm("supervisor")

    # Create all experts
    experts = create_all_experts(method=method)

    # Create supervisor
    supervisor = SupervisorAgent(
        llm=supervisor_llm,
        experts=experts,
        method=method,
        max_iterations=max_iterations,
    )

    return supervisor
