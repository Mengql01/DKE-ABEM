"""
Planner Expert Agent for TRNSYS DCK Generator.

Responsible for analyzing task descriptions and creating system plans.
This is the first expert called in the DCK generation workflow.

Key Responsibilities:
1. Analyze task description to understand system requirements
2. Identify system type (HVAC, solar, building, etc.)
3. Determine required component categories
4. Create a high-level component selection strategy

Design Principles:
- Use Memory/RAG tools to retrieve similar system designs
- No hardcoded system type detection
- LLM-driven analysis and planning
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage

from .base_expert import BaseExpert

logger = logging.getLogger(__name__)


class PlannerExpert(BaseExpert):
    """
    Planner Expert Agent.

    Analyzes task descriptions and creates system plans for DCK generation.
    """

    @property
    def name(self) -> str:
        return "planner"

    @property
    def system_prompt(self) -> str:
        # Dynamically read max_tool_calls from performance_config
        from ...core.performance_config import get_performance_config
        perf_config = get_performance_config()
        max_tool_calls = perf_config.planner_max_tool_calls

        return f"""Ⅰ. 【[CORE★★★★★] Role Definition】
You are a TRNSYS System Planning Expert. Your task is to analyze user's modeling task description and create a system planning scheme.

【【Quick Index — First Round — SystemMessage】】
Ⅰ. 【[CORE★★★★★] Role Definition】 — TRNSYS System Planning Expert, analyze task and create system planning scheme
Ⅱ. 【[CORE★★★★★] Workflow - FOLLOW STEP BY STEP】 — 3-step: retrieve similar designs → analyze key info → create planning scheme
Ⅲ. 【[CORE★★★★★] Output Format - ALL IN ENGLISH】 — JSON: system_type, system_description, component_categories, simulation_settings (SELECTIVE: only task-stated/implied control params), key_outputs, reference_systems
Ⅳ. 【[KEY★★★] Responsibilities】 — 4-step: analyze task → identify system type → determine component categories → create selection strategy
Ⅴ. 【[KEY★★★] Important Principles】 — Use tools not guess; no Type numbers; system-level only; English output
Ⅵ. 【[KEY★★★] Stop Condition】 — No limit on tool calls; no strict time requirement for JSON output; output JSON when prompt requirements are fully met; fallback to task description if no retrieval

------------------------------------------------------------
Ⅱ. 【[CORE★★★★★] Workflow - FOLLOW STEP BY STEP】
1. First use tools to retrieve similar system designs as reference
2. Analyze key information in task description:
   - System scale (equipment count, capacity)
   - System configuration (series/parallel, variable/fixed frequency)
   - Simulation time and step
   - Simulation-control parameters: tolerances, solver method, relaxation factors, DFQ,
     iteration/warning limits, debug switches — extract ONLY those the task mentions or implies
   - Required output variables
3. Based on retrieval results and task analysis, create planning scheme

------------------------------------------------------------
Ⅲ. 【[CORE★★★★★] Output Format - ALL IN ENGLISH】
Output in JSON format:
```json
{{
    "system_type": "System type (e.g., hvac, solar, building, cogeneration)",
    "system_description": "Brief system description",
    "component_categories": [
        {{
            "category": "Component category (e.g., weather, building, plotter, printer)",
            "count": 1,
            "description": "Purpose description"
        }}
    ],
    "simulation_settings": {{
        "start_time": 0,
        "stop_time": 8760,
        "time_step": 1
    }},
    "key_outputs": ["Key output variables"],
    "reference_systems": ["Referenced similar systems"]
}}
```

【[CORE★★★★★] simulation_settings — SELECTIVE OUTPUT RULE (read carefully)】
- The "simulation_settings" object above shows only the 3 time keys as an example.
- TRNSYS supports MANY more simulation-control parameters (full key list below).
- RULE: Output ONLY the keys that the Task Description explicitly states OR clearly implies.
  Do NOT output a key the task says nothing about — omit it, and the system fills it with the
  correct hardcoded default automatically. Never dump the whole list with default values.
- The task often describes the MEANING, not the number. Infer the numeric value from that meaning
  (see the inference hints below). When the task is silent about a parameter, leave it out.
- [CRITICAL] EXACT KEY NAMES — each key you emit MUST match the key list below CHARACTER-FOR-CHARACTER:
  same lowercase letters, same underscores, NO spaces, NO camelCase, NO renaming, NO abbreviation.
  Example: write "integration_tolerance" — NOT "Integration_Tolerance", "integrationTolerance",
  or "integration tolerance". A mismatched key cannot be read by the downstream code and the
  parameter is SILENTLY LOST. When unsure, copy the key verbatim from the list.
- [FORMAT] Emit each parameter as its OWN separate key on its OWN line in the JSON object. Every
  parameter is independent — never merge several parameters into one combined key or one line.

【Full key list — exact key names you MUST use; ONE key per line, copy verbatim】
- "start_time" (hours, default 0) → SIMULATION start time.
- "stop_time" (hours, default 8760) → SIMULATION stop time.
- "time_step" (hours, default 1) → SIMULATION time step.
- "integration_tolerance" (default 0.001) → TOLERANCES value 1, controls integration error.
- "convergence_tolerance" (default 0.001) → TOLERANCES value 2, controls input/output convergence.
    Smaller = more accurate but slower; larger = faster but coarser.
- "max_iterations" (default 30) → LIMITS value 1: iterations per time step before a WARNING.
- "max_warnings" (default 30) → LIMITS value 2: warnings before the run is terminated with an ERROR.
- "trace_limit" (default 30) → LIMITS value 3: iterations before non-converging components are traced.
- "dfq" (default 1) → DFQ differential-equation algorithm: 1=Modified-Euler, 2=non-self-starting
    Heun's, 3=fourth-order Adams.
- "width" (default 80) → WIDTH: characters per line of TRNSYS output.
- "solver_type" (default 0) → SOLVER method: 0=successive substitution (MUST be used for buildings
    and systems with thermal capacitance); 1=Powell's method (suited to systems with many non-linear
    equations / low capacitance, e.g. PV systems).
- "min_relaxation" (default 1) → SOLVER minimum relaxation factor.
- "max_relaxation" (default 1) → SOLVER maximum relaxation factor.
- "nan_check" (default 0) → NAN_CHECK debug switch: 0=off, 1=on (stop on NaN, for debugging).
- "overwrite_check" (default 0) → OVERWRITE_CHECK debug switch: 0=off, 1=on.
- "time_report" (default 0) → TIME_REPORT switch: 0=off, 1=on.
- "eqsolver" (default 0) → EQSOLVER equation-block update strategy: 0=TRNSYS-15-style blocking,
    1=update when a value changes beyond TOLERANCES, 2=treat equations as a component updated last.

【Meaning → value inference hints (use ONLY when the task points to it)】
- "high accuracy / high precision / tight tolerance / strict convergence" → smaller
    "integration_tolerance" and/or "convergence_tolerance" (e.g. 0.0001).
- "faster / coarse / loose tolerance" → larger tolerance (e.g. 0.01).
- "PV system / photovoltaic / many non-linear equations / low thermal capacity" → "solver_type": 1.
- "building / thermal mass / thermal capacitance / multi-zone building" → "solver_type": 0.
- "convergence is hard / oscillation / needs relaxation / damping" → set "min_relaxation" < 1
    (e.g. 0.5) while keeping "max_relaxation" at 1.
- "debug mode / diagnose NaN values" → "nan_check": 1.
- "check for output overwrite issues" → "overwrite_check": 1.
- "report runtime / timing" → "time_report": 1.
- explicit "iterations before warning" limit in the task → set "max_iterations".
- explicit "warnings before error" limit in the task → set "max_warnings".
- explicit "iterations before trace" limit in the task → set "trace_limit".
- Note: LIST/MAP/DEBUG cards are fixed by the assembler and are NOT configurable here — do not output them.

------------------------------------------------------------
Ⅳ. 【[KEY★★★] Responsibilities】
1. Analyze task description to understand the energy system to be modeled
2. Identify system type (e.g., HVAC, solar system, building thermal simulation, cogeneration)
3. Determine required component categories
4. Create component selection strategy

------------------------------------------------------------
Ⅴ. 【[KEY★★★] Important Principles】
- Use tools to retrieve relevant knowledge, do not guess
- Learn system design patterns from retrieval results
- Do NOT specify specific Type numbers in planning - that is component_selector's job
- Focus on system-level planning, not specific component parameter configuration
  (component PARAMETERS are parameter_configurator's job; however the global
  simulation_settings control cards ARE part of planning — fill them per the SELECTIVE OUTPUT RULE)
- For simulation_settings: output ONLY parameters the Task Description states or implies; omit the
  rest so the system applies correct defaults. Infer numeric values from descriptive wording.
- ALL output must be in ENGLISH - no Chinese characters
- For planner phase: ONLY use rag_search_similar_systems (do NOT use rag_search_global_for_dck_and_tmf)
- When calling rag_search_similar_systems, pass a SHORT English description (3-6 keywords/phrases) as task_description,
  NOT the full original task text. Example: "single zone building, PID heating cooling control"
- Do NOT use rag_search_global_for_dck_and_tmf in planning phase - component_selector will use it later with planned types

------------------------------------------------------------
Ⅵ. 【[KEY★★★] Stop Condition】
- No limit on the maximum number of tool calls; you may call tools as many times as needed to fulfill the prompt requirements
- No strict time requirement for JSON output; output the JSON result once you have completed all prompt requirements (self-judge when ready)
- If no relevant information retrieved, create planning based on task description directly
"""

    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the task and create a system plan.

        Args:
            state: Current state containing task_description.

        Returns:
            State updates with system_analysis and component_plan.
        """
        task_description = state.get("task_description", "")

        if not task_description:
            logger.error("No task description provided")
            return {
                "system_analysis": "Error: no task description provided",
                "component_plan": [],
                "status": "failed",
            }

        self.log_action("Planning started", f"Task: {task_description[:100]}...")

        # Create task message
        task_message = HumanMessage(content=f"""1. 【Task Instruction】
Please analyze the following TRNSYS modeling task and produce a system planning result:

【【Quick Index — First Round — HumanMessage】】
1. 【Task Instruction】 — Analyze the TRNSYS modeling task and produce a system planning result
2. 【Task Description】 — The user's modeling task description
3. 【Requirements】 — MANDATORY TOOL CALLING: retrieve similar designs → analyze key requirements → develop strategy → output JSON
4. 【Action】 — Begin the analysis

------------------------------------------------------------
2. 【Task Description】
{task_description}

------------------------------------------------------------
3. 【Requirements】 - MANDATORY TOOL CALLING
1. YOU MUST use rag_search_similar_systems tool to retrieve reference systems:
   - Extract 3-6 key English keywords/phrases from the task (e.g., "single zone building, PID control, heating cooling")
   - Call: rag_search_similar_systems(task_description="your keywords")
   - Analyze the retrieved reference designs to understand system patterns
2. Based on retrieval results and task analysis, identify:
   - System type (HVAC, solar, building, cogeneration, etc.)
   - Required component categories
   - Simulation settings (time/step PLUS any control parameters the task states or implies:
     tolerances, solver_type, relaxation, dfq, limits, debug switches — output ONLY those, omit
     the rest for defaults) and key outputs
3. Develop a component selection strategy
4. Output the planning result in JSON format

------------------------------------------------------------
4. 【Action】
Please begin the analysis. Start by extracting key keywords and calling rag_search_similar_systems.
""")

        try:
            # Invoke agent
            response = self.invoke_agent([task_message])

            # Extract result
            result_text = self.extract_result(response)

            # Parse JSON from result
            plan = self._parse_plan(result_text)

            self.log_action("Planning completed", f"Detected system type: {plan.get('system_type', 'unknown')}")

            return {
                "system_analysis": result_text,
                "component_plan": plan.get("component_categories", []),
                "simulation_settings": plan.get("simulation_settings", {}),
                "status": "planning_complete",
                "current_agent": "planner",
                "next_agent": "component_selector",
            }

        except Exception as e:
            logger.error(f"Planner error: {e}")
            return {
                "system_analysis": f"Planning failed: {str(e)}",
                "component_plan": [],
                "status": "failed",
            }

    def _parse_plan(self, result_text: str) -> Dict[str, Any]:
        """
        Parse the planning result to extract structured data.

        Args:
            result_text: Raw result text from LLM.

        Returns:
            Parsed plan dictionary.
        """
        # Try to extract JSON from the result
        try:
            # Look for JSON block
            import re
            json_match = re.search(r'```json\s*(.*?)\s*```', result_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))

            # Try to parse the entire text as JSON
            return json.loads(result_text)

        except json.JSONDecodeError:
            # If JSON parsing fails, create a basic structure
            logger.warning("Could not parse JSON from planner result")
            return {
                "system_type": "unknown",
                "system_description": result_text[:500],
                "component_categories": [],
                "simulation_settings": {},
                "key_outputs": [],
                "reference_systems": [],
            }

    def analyze_task(self, task_description: str) -> Dict[str, Any]:
        """
        Standalone method to analyze a task description.

        Args:
            task_description: The task description to analyze.

        Returns:
            Analysis result dictionary.
        """
        state = {"task_description": task_description}
        result = self.process(state)
        return {
            "system_analysis": result.get("system_analysis", ""),
            "component_plan": result.get("component_plan", []),
        }
