"""
Base Expert Agent for TRNSYS DCK Generator.

Provides the abstract base class for all expert agents.
Uses LangGraph create_react_agent for agent creation.

Key Design Principles:
- All experts inherit from BaseExpert
- Tool configuration based on method (Proposed vs B3)
- No hardcoded DCK generation logic
- LLM-driven decision making
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent

logger = logging.getLogger(__name__)


@dataclass
class ExpertConfig:
    """Configuration for an expert agent."""

    name: str
    role: str
    temperature: float = 0.3
    max_tokens: int = 8192
    tools: List[BaseTool] = field(default_factory=list)
    system_prompt: str = ""


# Temperature settings for different expert roles
EXPERT_TEMPERATURES = {
    "planner": 0.3,
    "component_selector": 0.5,
    "parameter_configurator": 0.3,
    "connection_generator": 0.3,
    "equation_generator": 0.2,
    "layout_designer": 0.5,
    "validator": 0.2,
    "supervisor": 0.5,
}


def create_expert_llm(expert_name: str, settings=None):
    """
    Create an LLM instance configured for a specific expert role.

    Args:
        expert_name: Name of the expert agent.
        settings: Optional settings object. If None, uses default settings.

    Returns:
        Configured LLM instance.
    """
    try:
        from ...core.llm_client import create_agent_llm
        from ...core.config import get_settings

        if settings is None:
            settings = get_settings()

        return create_agent_llm(expert_name, settings)

    except ImportError:
        # Fallback for testing
        logger.warning("Could not import llm_client, using fallback")
        from langchain_openai import ChatOpenAI

        temperature = EXPERT_TEMPERATURES.get(expert_name, 0.3)
        return ChatOpenAI(
            model="deepseek-chat",
            temperature=temperature,
            max_tokens=8192,
            max_retries=5,  # 自动重试5次
        )


class BaseExpert(ABC):
    """
    Abstract base class for expert agents.

    All expert agents inherit from this class and implement:
    - name: The expert's name
    - system_prompt: The expert's system prompt
    - process(): The main processing method

    The base class provides:
    - LangGraph agent creation via create_react_agent
    - Tool configuration based on method
    - State management utilities
    """

    REPAIR_TYPE_DIRECT = "direct_issue_fix"
    REPAIR_TYPE_ADAPTIVE = "indirect_adaptive_fix"

    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        tools: Optional[Sequence[BaseTool]] = None,
        method: str = "proposed",
        **kwargs,
    ):
        """
        Initialize the expert agent.

        Args:
            llm: LLM instance to use. If None, creates one based on expert role.
            tools: Tools to provide to the agent. If None, uses default tools for role.
            method: Method type ("proposed" or "b3_multiagent_rag")
            **kwargs: Additional configuration options.
        """
        self.method = method
        self._llm = llm
        self._tools = tools
        self._agent = None
        self._kwargs = kwargs

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the expert's name."""
        pass

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """Return the expert's system prompt."""
        pass

    @property
    def llm(self) -> BaseChatModel:
        """Get or create the LLM instance."""
        if self._llm is None:
            self._llm = create_expert_llm(self.name)
        return self._llm

    @property
    def tools(self) -> List[BaseTool]:
        """Get or create the tools list."""
        if self._tools is None:
            self._tools = self._get_default_tools()
        return list(self._tools)

    def _get_default_tools(self) -> List[BaseTool]:
        """Get default tools for this expert based on method."""
        try:
            from ..tools import get_tools_for_expert
            return get_tools_for_expert(self.name, self.method)
        except ImportError:
            logger.warning(f"Could not import tools for {self.name}")
            return []

    def get_agent(self) -> Callable:
        """
        Get or create the LangGraph agent.

        Returns:
            A callable LangGraph agent created with create_react_agent.
        """
        if self._agent is None:
            self._agent = create_react_agent(
                model=self.llm,
                tools=self.tools,
                prompt=self.system_prompt,  # Fixed: state_modifier -> prompt (LangGraph official API)
            )
        return self._agent

    _CANCEL_RESULT_TEMPLATE = {
        "user_cancelled": True,
        "next_agent": "FINISH",
        "status": "failed",
        "expert_output_status": "User manually interrupted the workflow",
    }

    async def __call__(self, state: Dict[str, Any], config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
        """
        Make the expert callable as a LangGraph node.

        Uses asyncio.shield to absorb CancelledError from the Runtime so that
        cancellation always goes through our post-execution check path, producing
        a graceful supervisor FINISH instead of an abrupt Runtime interruption.

        Args:
            state: The current state dictionary.
            config: Optional RunnableConfig injected by LangGraph runtime.

        Returns:
            Updated state dictionary.
        """
        run_id, thread_id = self._extract_config_ids(config)
        self._current_run_id = run_id
        self._current_thread_id = thread_id

        cancel_result = {**self._CANCEL_RESULT_TEMPLATE, "current_agent": self.name}

        # Check cancellation via registry + Runtime API
        if self._check_run_cancelled():
            logger.info("[%s] Run %s cancelled, skipping execution", self.name, run_id)
            return cancel_result
        # Also honour explicit state flag (set via State Edit / API)
        if state.get("user_cancelled", False):
            logger.info("[%s] User cancellation detected (state flag), skipping execution", self.name)
            return cancel_result

        loop = asyncio.get_event_loop()
        cancelled_during_process = False
        import concurrent.futures
        sync_fut = concurrent.futures.Future()
        ctx = contextvars.copy_context()

        def _run_process():
            try:
                sync_fut.set_result(ctx.run(self.process, state))
            except Exception as exc:
                sync_fut.set_exception(exc)

        loop.run_in_executor(None, _run_process)
        async_fut = asyncio.wrap_future(sync_fut)
        try:
            result = await asyncio.shield(async_fut)
        except asyncio.CancelledError:
            cancelled_during_process = True
            logger.info("[%s] CancelledError absorbed by shield, waiting for process() to finish", self.name)
            result = sync_fut.result(timeout=600)

        # Post-execution check: cancel may have arrived while process() was running
        if cancelled_during_process or self._check_run_cancelled():
            logger.info("[%s] Run %s cancelled after process() completed, discarding result", self.name, run_id)
            return cancel_result
        return result

    @abstractmethod
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the state and return updates.

        This is the main method that each expert must implement.
        It should:
        1. Extract relevant information from state
        2. Use tools to gather information
        3. Generate output using LLM
        4. Return state updates

        Args:
            state: The current state dictionary.

        Returns:
            Dictionary with state updates.
        """
        pass

    @staticmethod
    def _extract_config_ids(config) -> tuple:
        """Extract (run_id, thread_id) from LangGraph RunnableConfig."""
        if not config:
            return "", ""
        run_id = config.get("run_id", "")
        if not run_id:
            run_id = (config.get("configurable") or {}).get("run_id", "")
        thread_id = (config.get("configurable") or {}).get("thread_id", "")
        return (str(run_id) if run_id else "", str(thread_id) if thread_id else "")

    def _check_run_cancelled(self) -> bool:
        run_id = getattr(self, "_current_run_id", "")
        if not run_id:
            return False
        from ...core.cancellation import is_run_cancelled, check_run_status_via_api
        if is_run_cancelled(run_id):
            return True
        thread_id = getattr(self, "_current_thread_id", "")
        return check_run_status_via_api(thread_id, run_id)

    def invoke_agent(
        self,
        messages: List[Union[HumanMessage, AIMessage, SystemMessage]],
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Invoke the agent with messages.

        Args:
            messages: List of messages to send to the agent.
            **kwargs: Additional arguments for the agent.

        Returns:
            Agent response.
        """
        agent = self.get_agent()
        from ...core.performance_config import get_performance_config
        perf_config = get_performance_config()

        config = {
            "recursion_limit": perf_config.heavy_expert_recursion_limit
            if self.name in ("connection_generator", "parameter_configurator")
            else perf_config.expert_recursion_limit,
            "tags": [f"expert-{self.name}", self.method],
            "metadata": {
                "expert_name": self.name,
                "method": self.method,
            },
            "run_name": f"Expert-{self.name}",
        }
        result = agent.invoke({"messages": messages}, config=config, **kwargs)
        return result

    def invoke_agent_with_prompt(
        self,
        messages: List[Union[HumanMessage, AIMessage, SystemMessage]],
        prompt_override: Optional[str] = None,
        tools_override: Optional[List[BaseTool]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Invoke agent with optional prompt override (used for repair-round wrapper prompts).

        Args:
            messages: List of messages to pass to the agent
            prompt_override: Optional system prompt override
            tools_override: Optional tools list override (if None, uses self.tools; if [], disables tools)
            **kwargs: Additional arguments to pass to agent.invoke()

        Note:
            When tools_override is explicitly provided (including empty list []), it is ALWAYS respected
            regardless of prompt_override value. This ensures tool disabling is reliable.
            Only when both prompt_override and tools_override are None/default will invoke_agent() be used.
        """
        # Determine tools to use - ALWAYS process tools_override if explicitly provided
        # This is the critical fix: tools_override must be respected even when prompt_override is None
        tools_to_use = tools_override if tools_override is not None else self.tools

        # If no prompt override and tools are default, use invoke_agent() for efficiency
        # This optimization avoids creating a temp_agent when not needed
        if not prompt_override and tools_override is None:
            return self.invoke_agent(messages, **kwargs)

        # Create temp agent with specified prompt and tools
        # This path is taken when:
        # 1. prompt_override is provided (repair prompt), OR
        # 2. tools_override is explicitly provided (including [])
        temp_agent = create_react_agent(
            model=self.llm,
            tools=tools_to_use,
            prompt=prompt_override or self.system_prompt,
        )

        from ...core.performance_config import get_performance_config
        perf_config = get_performance_config()

        # Determine if this is a repair prompt
        is_repair_prompt = bool(prompt_override)

        # Build tags dynamically based on actual parameters
        tags = [f"expert-{self.name}", self.method]
        if is_repair_prompt:
            tags.append("repair-prompt-wrapper")
        if tools_override is not None:
            # Add informative tag about tool override status
            tags.append(f"tools-override-count-{len(tools_to_use)}")

        config = {
            "recursion_limit": perf_config.heavy_expert_recursion_limit
            if self.name in ("connection_generator", "parameter_configurator")
            else perf_config.expert_recursion_limit,
            "tags": tags,
            "metadata": {
                "expert_name": self.name,
                "method": self.method,
                "repair_prompt": is_repair_prompt,
                "tools_override_specified": tools_override is not None,
                "tools_count": len(tools_to_use),
            },
            "run_name": f"Expert-{self.name}" + ("-Repair" if is_repair_prompt else ""),
        }
        result = temp_agent.invoke({"messages": messages}, config=config, **kwargs)
        return result

    def create_task_message(self, state: Dict[str, Any]) -> HumanMessage:
        """
        Create a task message from the current state.

        Subclasses can override this to customize the task message.

        Args:
            state: The current state dictionary.

        Returns:
            HumanMessage with the task description.
        """
        task_description = state.get("task_description", "")
        return HumanMessage(content=f"Please process the following task:\n\n{task_description}")

    def is_repair_round(self, state: Dict[str, Any]) -> bool:
        """Return True only for non-first-round repair executions.

        Strict rule: validator_round > 0 means at least one validator boundary has passed,
        so experts are in repair context. This avoids first-round misclassification caused by
        transient state flags.
        """
        return state.get("validator_round", 0) > 0

    def get_agent_last_output(self, state: Dict[str, Any], agent_name: Optional[str] = None) -> Dict[str, Any]:
        """Get previous raw-output snapshot for the current (or specified) agent."""
        key = agent_name or self.name
        agent_outputs = state.get("agent_last_outputs", {}) or {}
        return agent_outputs.get(key, {}) or {}

    def build_agent_output_snapshot_updates(
        self,
        state: Dict[str, Any],
        raw_output: str,
        agent_name: Optional[str] = None,
        json_str: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Persist last raw output snapshot for this agent (used by next repair round input).

        Args:
            state: Current state
            raw_output: Full LLM output text (for backward compatibility)
            agent_name: Agent name override
            json_str: Pre-extracted JSON string from _parse method (preferred)
        """
        key = agent_name or self.name
        agent_last_outputs = dict(state.get("agent_last_outputs", {}) or {})

        # Prefer pre-extracted JSON string from _parse method
        snapshot_json = json_str if json_str is not None else raw_output

        agent_last_outputs[key] = {
            "validator_round": state.get("validator_round", 0),
            "raw_output": snapshot_json,
        }
        return {"agent_last_outputs": agent_last_outputs}

    def cache_first_round_output(
        self,
        state: Dict[str, Any],
        raw_output: str,
        agent_name: Optional[str] = None,
        json_str: Optional[str] = None,
    ) -> None:
        """Persist first-round raw JSON outputs on the in-memory expert instance only.

        This keeps first-round cache available for later repair promotion without
        exposing the cache field in LangGraph state input/output.
        """
        key = agent_name or self.name
        cache = dict(getattr(self, "_first_round_output_cache", {}) or {})
        cache[key] = {
            "validator_round": state.get("validator_round", 0),
            "raw_output": json_str if json_str is not None else raw_output,
        }
        self._first_round_output_cache = cache

    def get_first_round_output_cache(self, agent_name: Optional[str] = None) -> Dict[str, Any]:
        """Get first-round raw-output cache from in-memory expert instance."""
        key = agent_name or self.name
        cache = dict(getattr(self, "_first_round_output_cache", {}) or {})
        return cache.get(key, {}) or {}

    def _derive_agent_baseline_output(self, state: Dict[str, Any], agent_name: str) -> Dict[str, Any]:
        """Fallback baseline output when explicit snapshot is unavailable."""
        if agent_name == "component_selector":
            return {"components": state.get("components", [])}
        if agent_name == "equation_generator":
            return {
                "equations": state.get("equations", []),
                "constants": state.get("constants", []),
            }
        if agent_name == "parameter_configurator":
            return {"components": state.get("components", [])}
        if agent_name == "connection_generator":
            return {
                "components": state.get("components", []),
                "connections": state.get("connections", []),
                "equations": state.get("equations", []),
            }
        if agent_name == "layout_designer":
            return {
                "components": state.get("components", []),
                "equations": state.get("equations", []),
                "layout_styles": state.get("layout_styles", []),
                "link_style_lines": state.get("link_style_lines", []),
            }
        return {}

    def build_repair_context(self, state: Dict[str, Any], agent_name: Optional[str] = None) -> Dict[str, Any]:
        """Build normalized repair context consumed by experts in repair rounds."""
        key = agent_name or self.name
        agent_last_output = self.get_agent_last_output(state, key)
        if not agent_last_output:
            cached_output = self.get_first_round_output_cache(key)
            if cached_output:
                agent_last_output = cached_output
            else:
                agent_last_output = {
                    "validator_round": state.get("validator_round", 0),
                    "raw_output": "",
                }

        current_round_inputs = {
            "components": state.get("components", []),
            "connections": state.get("connections", []),
            "equations": state.get("equations", []),
            "constants": state.get("constants", []),
            "layout_styles": state.get("layout_styles", []),
            "link_style_lines": state.get("link_style_lines", []),
        }

        return {
            "current_agent": key,
            "last_validator_snapshot": state.get("last_validator_snapshot", {}) or {},
            "current_round_repair_notes": state.get("current_round_repair_notes", []) or [],
            "agent_last_output": agent_last_output,
            "repair_execution_chain": state.get("repair_execution_chain", []) or [],
            "repair_targets": state.get("repair_targets", []) or [],
            "validator_round": state.get("validator_round", 0),
            "current_round_inputs": current_round_inputs,
        }

    def wrap_prompt_with_repair_instructions(self, original_prompt: str) -> str:
        """
        Wrap first-round prompt with repair-round instructions.

        IMPORTANT:
        - Keep the original prompt unchanged as inner context.
        - Repair instructions are an outer wrapper, active in repair rounds only.
        """
        repair_wrapper = """
【【Quick Index — REPAIR Round — SystemMessage】】
Ⅰ. 【[CORE★★★★★] Repair Role Definition】
You are working in a repair round, not an initial generation round. Your job is to review the previous validation result, understand whether your part actually needs updates in this round, and only then make targeted edits if needed.

【【Quick Index — Repair Round】】
Ⅰ. 【[CORE★★★★★] Repair Role Definition】 — Repair round context: review validation result, decide if updates needed, make targeted edits
Ⅱ. 【[CORE★★★★★] MANDATORY REPAIR Workflow - FOLLOW STEP BY STEP】 — 4-step: reconfirm role → review context & decide → build edit plan → produce updated result
Ⅲ. 【[CORE★★★★★] Output Format in Repair Round - ALL IN ENGLISH】 — Normal output + repair_summary (max 100 chars) + repair_detail (type/units/aspect/how)

------------------------------------------------------------
Ⅱ. 【[CORE★★★★★] MANDATORY REPAIR Workflow - FOLLOW STEP BY STEP】
Follow in order:

Step 1) Reconfirm your role and output contract
- Read [ORIGINAL FIRST-ROUND PROMPT] only to understand your responsibility boundaries, constraints, tools, and required output format.
- Do not treat the first-round prompt as an instruction to regenerate everything from scratch.
- Keep in mind that this round is about reviewing and updating previous work, not rebuilding it from zero.

Step 2) Review the repair context and decide whether changes are needed
You must carefully review the following repair-context information:
- Previous validation result
- Validation errors to address
- Failure analysis and likely causes
- Updates already made earlier in this repair chain
- Your previous output (editing baseline)
- The current business state relevant to this round

Then decide:
- Whether your part actually needs changes in this round.
- Whether your work, if needed, is mainly:
  - direct_issue_fix: directly addressing the validation failure.
  - indirect_adaptive_fix: adapting your output because earlier experts changed something upstream.

Step 3) If changes are needed, build an edit plan against your previous output
- Your editing baseline is: Your previous output (editing baseline).
- Analyze the failure before editing. Do not use blind trial-and-error.
- Identify:
  - What must change,
  - What may need adjustment,
  - What must remain unchanged.
- You may use tools when needed to gather repair evidence, but do not fall back to a full regeneration workflow.
- Keep the first-round output structure, but apply focused edits to the baseline instead of recreating everything.

Step 4) Produce the updated result
- If changes are required, output a revised result that preserves the first-round structure while making only the necessary targeted updates.
- If no changes are required, keep the baseline logically unchanged and only make the minimum consistency adjustments required by earlier updates in this repair chain or by the current business state.

------------------------------------------------------------
Ⅲ. 【[CORE★★★★★] Output Format in Repair Round - ALL IN ENGLISH】
In addition to your normal structured output, you must include two extra fields:
- "repair_summary": a concise one-line summary (max 100 chars) of what was updated or why no update was needed
- "repair_detail": a detailed explanation that must include:
  a) whether this is a "direct_issue_fix" or "indirect_adaptive_fix"
  b) which specific unit(s), block(s), or section(s) were touched
  c) what aspect changed (parameters, inputs, equations, connections, layout, etc.)
  d) how it changed (added, removed, reordered, updated value, kept unchanged, etc.)

If no changes are required, set repair_summary="No changes required" and repair_detail="Reviewed the repair context and kept the previous output as the editing baseline with no substantive changes."
[/REPAIR MODE OUTER INSTRUCTIONS]
""".strip()
        return f"{repair_wrapper}\n\n[ORIGINAL FIRST-ROUND PROMPT]\n{original_prompt}\n[/ORIGINAL FIRST-ROUND PROMPT]"

    def extract_repair_info_from_output(
        self,
        output: Dict[str, Any],
        fallback_summary: str = "",
        fallback_detail: str = "",
    ) -> tuple[str, str]:
        """
        Extract repair_summary and repair_detail from LLM output.

        Args:
            output: LLM structured output dictionary
            fallback_summary: Fallback summary if not found in output
            fallback_detail: Fallback detail if not found in output

        Returns:
            Tuple of (repair_summary, repair_detail)
        """
        summary = output.get("repair_summary", "").strip()
        detail = output.get("repair_detail", "").strip()

        # Use fallback only if LLM didn't provide the fields
        if not summary:
            summary = fallback_summary
        if not detail:
            detail = fallback_detail

        return summary, detail

    def build_repair_note(
        self,
        summary: str,
        detail: str,
        repair_type: str,
        changed: bool,
        state: Dict[str, Any],
        agent_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build standardized repair note record for repair rounds."""
        key = agent_name or self.name
        normalized_type = repair_type if repair_type in {
            self.REPAIR_TYPE_DIRECT,
            self.REPAIR_TYPE_ADAPTIVE,
        } else self.REPAIR_TYPE_ADAPTIVE

        return {
            "repair_agent": key,
            "repair_type": normalized_type,
            "repair_summary": summary.strip() if isinstance(summary, str) else "",
            "repair_detail": detail.strip() if isinstance(detail, str) else "",
            "no_change": not changed,
            "validator_round": state.get("validator_round", 0),
        }

    def build_repair_round_state_updates(
        self,
        state: Dict[str, Any],
        raw_output: str,
        repair_note: Dict[str, Any],
        agent_name: Optional[str] = None,
        json_str: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build shared repair-round state updates to merge into expert outputs.

        Args:
            state: Current state
            raw_output: Full LLM output text (for backward compatibility)
            repair_note: Repair note dictionary
            agent_name: Agent name override
            json_str: Pre-extracted JSON string from _parse method (preferred)
        """
        key = agent_name or self.name

        agent_last_outputs = dict(state.get("agent_last_outputs", {}) or {})
        # Prefer pre-extracted JSON string from _parse method
        snapshot_json = json_str if json_str is not None else raw_output
        agent_last_outputs[key] = {
            "validator_round": state.get("validator_round", 0),
            "raw_output": snapshot_json,
        }

        current_round_repair_notes = list(state.get("current_round_repair_notes", []) or [])
        current_round_repair_notes.append(repair_note)

        repair_notes_history = list(state.get("repair_notes_history", []) or [])
        repair_notes_history.append(repair_note)

        return {
            "agent_last_outputs": agent_last_outputs,
            "current_round_repair_notes": current_round_repair_notes,
            "repair_notes_history": repair_notes_history,
        }

    def _format_state_field_as_json_block(self, value: Any) -> str:
        """
        Render a state field as a unified, model-friendly ```json ... ``` block.

        Goal: make all upstream state fields share an identical visual and
        structural form inside the human prompt, so the LLM does not need to
        switch between two different presentation styles for upstream data.

        Behavior:
        - dict / list: serialize with ``json.dumps(indent=2, ensure_ascii=False)``
          and wrap in a ```json fenced code block.
        - str: best-effort parse the string as JSON (either embedded inside an
          existing ```json fence or as a whole), then re-serialize in the same
          unified form. If parsing fails at every step, fall back to the
          ORIGINAL string untouched so no upstream information is ever lost.
        - other types: convert via ``str()`` and return unfenced (extremely
          unlikely path, kept only as a safety net).

        Args:
            value: The raw state field value (str / dict / list / other).

        Returns:
            A string ready to be interpolated into the prompt template.
        """
        # Structured object path: dict or list
        if isinstance(value, (dict, list)):
            try:
                serialized = json.dumps(value, ensure_ascii=False, indent=2)
                return f"```json\n{serialized}\n```"
            except (TypeError, ValueError):
                return str(value)

        # Raw string path
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return value

            # 1) Try to extract a ```json ... ``` fenced block from the text.
            fence_match = re.search(r'```json\s*(.*?)\s*```', stripped, re.DOTALL)
            if fence_match:
                candidate = fence_match.group(1).strip()
                try:
                    parsed = json.loads(candidate)
                    serialized = json.dumps(parsed, ensure_ascii=False, indent=2)
                    return f"```json\n{serialized}\n```"
                except (TypeError, ValueError):
                    pass  # fall through to the next strategy

            # 2) Try to parse the entire string as JSON.
            try:
                parsed = json.loads(stripped)
                serialized = json.dumps(parsed, ensure_ascii=False, indent=2)
                return f"```json\n{serialized}\n```"
            except (TypeError, ValueError):
                pass

            # 3) Fallback: return the original string unmodified so we never
            # lose information from upstream.
            return value

        # Safety net for unexpected types
        return str(value)

    def build_repair_context_text(self, repair_context: Dict[str, Any]) -> str:
        """Render a concise repair brief for expert task messages."""
        validator_snapshot = repair_context.get("last_validator_snapshot", {}) or {}
        current_round_repair_notes = repair_context.get("current_round_repair_notes", []) or []
        agent_last_output = repair_context.get("agent_last_output", {}) or {}

        validation_result = validator_snapshot.get("validation_result", "unknown")
        validation_errors = validator_snapshot.get("validation_errors", []) or []
        failure_analysis = validator_snapshot.get("error_analysis", {}) or {}
        previous_output_round = agent_last_output.get("validator_round", repair_context.get("validator_round", 0))
        editing_baseline = agent_last_output.get("raw_output", "")

        return (
            "[Repair Brief]\n"
            f"Previous validation result: {validation_result}\n"
            f"Validation errors to address:\n{self._format_state_field_as_json_block(validation_errors)}\n"
            f"Failure analysis and likely causes:\n{self._format_state_field_as_json_block(failure_analysis)}\n"
            f"Updates already made earlier in this repair chain:\n{self._format_state_field_as_json_block(current_round_repair_notes)}\n"
            f"Your previous output (editing baseline, produced in validation round {previous_output_round}):\n{self._format_state_field_as_json_block(editing_baseline)}"
        )

    def extract_result(self, response: Dict[str, Any]) -> str:
        """
        Extract the result from agent response.

        Args:
            response: Agent response dictionary.

        Returns:
            Extracted result string.
        """
        messages = response.get("messages", [])
        if messages:
            last_message = messages[-1]
            # Handle both message objects and dictionaries
            if hasattr(last_message, "content"):
                content = last_message.content
            elif isinstance(last_message, dict) and "content" in last_message:
                content = last_message["content"]
            else:
                return ""

            if isinstance(content, str):
                return content
            if isinstance(content, list):
                text_parts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                    elif isinstance(block, str):
                        text_parts.append(block)
                return "\n".join([p for p in text_parts if p])
            return str(content)
        return ""

    def log_action(self, action: str, details: str = ""):
        """Log an action taken by the expert."""
        logger.info(f"[{self.name}] {action}: {details}")

    @staticmethod
    def _ensure_unit_number_first(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Ensure unit_number is the first field in all dictionaries.

        Args:
            items: List of dictionaries that may contain unit_number field.

        Returns:
            List of dictionaries with unit_number as first field (if present).
        """
        reordered = []
        for item in items:
            if isinstance(item, dict) and "unit_number" in item:
                # Rebuild dict with unit_number first
                new_item = {"unit_number": item["unit_number"]}
                for k, v in item.items():
                    if k != "unit_number":
                        new_item[k] = v
                reordered.append(new_item)
            else:
                # No unit_number or not a dict, keep as is
                reordered.append(item)
        return reordered

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, method={self.method})"


class ExpertResult:
    """Container for expert processing results."""

    def __init__(
        self,
        success: bool,
        output: Any = None,
        error: Optional[str] = None,
        state_updates: Optional[Dict[str, Any]] = None,
    ):
        self.success = success
        self.output = output
        self.error = error
        self.state_updates = state_updates or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "state_updates": self.state_updates,
        }

    @classmethod
    def success_result(cls, output: Any, state_updates: Dict[str, Any] = None):
        """Create a success result."""
        return cls(success=True, output=output, state_updates=state_updates)

    @classmethod
    def error_result(cls, error: str):
        """Create an error result."""
        return cls(success=False, error=error)
