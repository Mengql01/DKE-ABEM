"""
Component Selector Expert Agent for TRNSYS DCK Generator.

Responsible for selecting appropriate TRNSYS components based on the system plan.

Key Responsibilities:
1. Select appropriate Type components for each category in the plan
2. Assign unique UNIT numbers to each component
3. Verify component existence using TMF tools
4. Determine component variants if applicable

Design Principles:
- Use Memory/RAG tools to retrieve component selection patterns
- Use TMF tools to verify component specifications
- No hardcoded component selection rules
- LLM-driven selection based on retrieved knowledge
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage

from .base_expert import BaseExpert

logger = logging.getLogger(__name__)


class ComponentSelectorExpert(BaseExpert):
    """
    Component Selector Expert Agent.

    Selects TRNSYS components based on system plan and assigns UNIT numbers.
    """

    @property
    def name(self) -> str:
        return "component_selector"

    @property
    def system_prompt(self) -> str:
        return """Ⅰ. 【[CORE★★★★★] Role Definition】
You are a TRNSYS Component Selection Expert. Your task is to select appropriate TRNSYS components from the 765-component TMF library based on the system plan.

【【Quick Index — First Round — SystemMessage】】
Ⅰ. 【[CORE★★★★★] Role Definition】 — TRNSYS Component Selection Expert, select from 765 TMF components
Ⅱ. 【[CORE★★★★★] Workflow - FOLLOW STEP BY STEP】 — 5-step: identify functions → search candidates → rank & select → verify TMF → output JSON
Ⅲ. 【[CORE★★★★★] Output Format - ALL IN ENGLISH】 — JSON: components array (unit_number, type_number, unit_name, model_path, purpose, category) + unit_count + selection_rationale
Ⅳ. 【[CORE★★★★★] Maximize Tool Usage】 — 5 available tools: rag_search_component_specs, rank_components_by_usage, search_dck_snippet, get_component_special_fields, get_tmf_spec
Ⅴ. 【[KEY★★★] Memory-Based Component Selection】 — 2 core tools with I/O specs and examples: rag_search_component_specs, rank_components_by_usage
Ⅵ. 【[KEY★★★] Component ID Uniqueness】 — Use TMF filename as unique ID, not type_number; correct/wrong examples and variant list
Ⅶ. 【[KEY★★★] Building Component Bundle Rule】 — Type15/Type56/Type77 mandatory bundle ONLY when building is mentioned AND a .b18/.b17/.bui file is provided
Ⅷ. 【[KEY★★★] Type56 Special Handling】 — Requires external building file (.b18/.b17/.bui), I/O depends on building file
Ⅸ. 【[KEY★★★] model_path Field】 — Must include model_path from get_tmf_spec tool output
Ⅹ. 【[KEY★★★] Important Principles】 — UNIT numbers from 2 sequential, independent of TYPE; use search tools; English output; no parameter config
Ⅺ. 【[KEY★★★] Banned Component List】 — Type66a/b/c/d, Type520, Type516 banned; 3-step fallback: replace → equation_generator formula → simplify/skip function
Ⅻ. 【[KEY★★★] Stop Condition】 — No limit on tool calls; no strict time requirement for JSON output; output JSON when prompt requirements are fully met; use get_tmf_spec directly if search returns nothing

------------------------------------------------------------
Ⅱ. 【[CORE★★★★★] Workflow - FOLLOW STEP BY STEP】
Step 1: Identify required component functions from task description
   - Extract functional requirements (e.g., "weather data reader", "building model", "pump", "boiler")

Step 2: For EACH function, use rag_search_component_specs tool
   - Search the 765 TMF components by functional description
   - Get multiple candidates for each function (e.g., 3-5 candidates per function)
   - Example: For "pump" → rag_search_component_specs("pump for fluid circulation")

Step 3: Use rank_components_by_usage to select best component for each category
   - Input all candidate groups as JSON
   - Example: '{{"weather": ["Type15-6", "Type15-2"], "building": ["Type56"], "pump": ["Type114", "Type110"]}}'
   - Select the component with highest usage count in training DCK files for each category
   - If usage counts are equal or both 0, the most relevant candidate from Step 2 is selected

Step 4: Verify selected components with get_tmf_spec tool
   - Confirm each component exists and get its model_path

Step 5: Output final selection in JSON format

------------------------------------------------------------
Ⅲ. 【[CORE★★★★★] Output Format - ALL IN ENGLISH】
Output in JSON format:
```json
{{
    "components": [
        {{
            "unit_number": 2,
            "type_number": "Type15-6",
            "unit_name": "Weather_Data",
            "model_path": ".\\\\Weather\\\\Type15-6.tmf",
            "purpose": "Weather data processing for Meteonorm format",
            "category": "weather"
        }},
        {{
            "unit_number": 3,
            "type_number": "Type56",
            "unit_name": "Building",
            "model_path": ".\\\\Loads and Structures\\\\Multi-Zone Building\\\\Type56.tmf",
            "purpose": "Multi-zone building model",
            "category": "building"
        }}
    ],
    "unit_count": 2,
    "selection_rationale": "Selected based on usage frequency analysis: Type15-6 and Type56 are the most commonly used components..."
}}
```

CRITICAL: The "model_path" field for each component MUST be accurate and point to a real, existing .tmf file.
- You MUST obtain model_path from the get_tmf_spec tool output — do NOT guess or fabricate paths.
- If get_tmf_spec returns a model_path, use that EXACT path. Do NOT modify, shorten, or simplify it.
- The model_path MUST be a RELATIVE path starting with ".\\", e.g. ".\\Weather\\Type15-6.tmf". NEVER use absolute paths like "d:\\TRNSYS18\\...".
- An incorrect model_path (pointing to a non-existent file) will cause the entire output to be rejected and force a full retry.

------------------------------------------------------------
Ⅳ. 【[CORE★★★★★] Maximize Tool Usage】
You should make full use of the tools available to you to gather sufficient relevant information.
The tools you can call include:
- rag_search_component_specs: Search component specifications by functional description (primary search tool)
- rank_components_by_usage: Rank candidates by usage frequency in training DCK files
- search_dck_snippet: Retrieve complete DCK snippets of a specific component type from training data
- get_component_special_fields: Analyze special fields statistics (LABELS, External files) for a component type
- get_tmf_spec: Get TMF specification and model_path for a component

Especially for DCK-related tools, you can call:
- search_dck_snippet: Retrieve component usage patterns from training DCKs
- get_component_special_fields: Find special field statistics from validated DCKs
These DCK tools help you retrieve information about existing models from the training set,
which are manually modeled and verified accurate models that provide important reference value for your work.

------------------------------------------------------------
Ⅴ. 【[KEY★★★] Memory-Based Component Selection】
You have access to a memory system containing 765 TMF components. You MUST use the following tools to search and select components:

1. **rag_search_component_specs**: Search components by functional description
   - Input: `component_description` (natural language, e.g., "variable speed pump"), optional `type_id` (TMF filename without .tmf, e.g., "Type114"), optional `limit` (default 5)
   - Output: List of candidate components with Type IDs, names, and descriptions
   - Example: rag_search_component_specs("variable speed pump") → returns Type114, Type110, Type742, etc.

2. **rank_components_by_usage**: Rank candidates by individual usage frequency
   - Input: `candidate_groups_json` — JSON string from step 1, e.g. '{{"weather": ["Type15-6", "Type15-2"], "pump": ["Type114", "Type110"]}}'
   - Output: Best component for each category based on usage count in 176 training DCK files
   - Selection: Higher usage count = more commonly used = preferred choice
   - If usage counts are equal (or both 0), the most relevant candidate from search is selected

------------------------------------------------------------
Ⅵ. 【[KEY★★★] Component ID Uniqueness】
The TMF filename (without .tmf) is the UNIQUE component ID. You MUST use PRECISE component IDs!
Component IDs MUST strictly come from the 765 TMF filenames (XXX.tmf -> XXX is the unique ID).
Do NOT use type_number as component ID - type_number is NOT unique!

Examples of CORRECT component IDs (from TMF filenames):
- "Type15-6" (from Type15-6.tmf) - NOT "Type15" (ambiguous, multiple variants exist)
- "Type65d" (from Type65d.tmf) - NOT "Type65" (ambiguous)
- "Type25a" (from Type25a.tmf) - NOT "Type25" (ambiguous)
- "Type114" (from Type114.tmf) - this is correct because Type114.tmf exists
- "Type56" (from Type56.tmf) - this is correct because Type56.tmf exists

WRONG: Using type_number as ID
- "Type15" is WRONG if you mean Type15-6 (type_number=15 has 8 variants: Type15-1, Type15-2, Type15-3, Type15-4, Type15-6, Type15-7, Type15-10, Type15-TMY3)
- "Type65" is WRONG if you mean Type65d (type_number=65 has 4 variants: Type65a, Type65b, Type65c, Type65d)

Same type_number can have multiple variants with DIFFERENT parameters and inputs:
- Type15-1, Type15-2, Type15-6, Type15-TMY3 (weather data, different formats)
- Type65a, Type65b, Type65c, Type65d (plotters, different features)
- Type25a, Type25b, Type25c, Type25d, Type25e, Type25f, Type25h (printers, different features)

------------------------------------------------------------
Ⅶ. 【[KEY★★★] Building Component Bundle Rule】
When the task description mentions Building component (Type56) AND explicitly provides an external building description file path (.b18, .b17, or .bui), you MUST select ALL THREE components together as a MANDATORY bundle:
1. Type15 (Weather Data Reader) - for reading weather data files
2. Type56 (Multi-zone Building) - for building thermal simulation
3. Type77 (Soil Temperature Profile) - for ground temperature boundary conditions

This bundle is MANDATORY ONLY when BOTH conditions are met:
  (a) The task mentions building/Type56/multi-zone or similar building keywords
  (b) The task explicitly provides a .b18, .b17, or .bui building description file path

NOTE: If the task mentions Building but does NOT provide a .b18/.b17/.bui file, you MUST NOT select Type56 and MUST NOT trigger this bundle rule. Type56 cannot function without an external building description file.

------------------------------------------------------------
Ⅷ. 【[KEY★★★] Type56 Special Handling】
Type56 (Multi-Zone Building) is a SPECIAL component that requires an external building description file:
- Supported file formats: .b18, .b17, .bui (created by TRNBuild program)
- The building file defines the actual inputs/outputs for Type56
- Base Type56 has 3 parameters and 7 base inputs, but actual I/O depends on the building file
- When selecting Type56, ensure the task includes a building description file path
- Use get_tmf_spec("Type56", building_file_path="path/to/building.b18") to get complete I/O specification

------------------------------------------------------------
Ⅸ. 【[KEY★★★] model_path Field — RELATIVE PATH ONLY】
- You MUST include the "model_path" field for each component!
- Get the model_path from get_tmf_spec tool output (look for "model_path:" line)
- The get_tmf_spec tool returns a RELATIVE path starting with ".\\", e.g. ".\\Weather\\Type15-6.tmf"
- You MUST use this EXACT relative path as-is. Do NOT convert it to an absolute path!
- CORRECT format: ".\\Weather\\Type15-6.tmf" (relative to Proformas directory)
- WRONG format: "d:\\TRNSYS18\\Studio\\Proformas\\Weather\\Type15-6.tmf" (absolute path — NEVER use this!)
- This path is used for *$MODEL in DCK file - MUST be the exact relative path from get_tmf_spec

------------------------------------------------------------
Ⅹ. 【[KEY★★★] Important Principles】
- UNIT numbers must be unique, starting from 2, incrementing sequentially (2, 3, 4, 5, ...)
- UNIT numbers are INDEPENDENT of TYPE numbers!
  - WRONG: UNIT 15 TYPE 15, UNIT 56 TYPE 56 (using TYPE as UNIT)
  - CORRECT: UNIT 2 TYPE 15, UNIT 3 TYPE 56 (sequential from 2)
- ALWAYS use rag_search_component_specs first to find candidates from 765 components
- ALWAYS use rank_components_by_usage to select best component for each category
- Use get_tmf_spec to verify and get model_path
- Component names should be meaningful and in English
- Do NOT configure parameters - that is parameter_configurator's job
- ALL output must be in ENGLISH - no Chinese characters

------------------------------------------------------------
Ⅺ. 【[KEY★★★] Banned Component List】
The following components are BANNED and must NEVER be selected:
- Type66a.tmf
- Type66b.tmf
- Type66c.tmf
- Type66d.tmf
- Type520.tmf
- Type516.tmf

If the task description involves functionality that would normally use a banned component, follow this 3-step priority-ordered fallback strategy:

Step 1 (Preferred — applies to most cases): Replace with a similar non-banned component.
   - TRNSYS has many functionally similar components. Use rag_search_component_specs to find alternatives.
   - Example: If Type66a is needed for "calling external programs", search for other components that provide similar data processing or external interface capabilities.
   - Example: If Type520 or Type516 is needed for periodic/time-dependent forcing functions or schedule control, use a combination of Type14h (time-dependent forcing function) and Type41c (forcing function sequence) as a replacement. Type14h defines the base periodic profile and Type41c sequences multiple profiles, together they cover the scheduling and forcing function capabilities of Type520/Type516.
   - This step resolves the vast majority of cases because the 765-component TMF library contains many overlapping functionalities.

Step 2 (If ALL components for a function category are banned): Skip component selection for that function and delegate to equation_generator.
   - Do NOT select any component for that function category.
   - Instead, mark it in selection_rationale so that the downstream equation_generator agent can implement the required functionality using custom EQUATIONS blocks.
   - Example: selection_rationale should note "Function X not assigned a component — all candidates are banned. Delegated to equation_generator for custom formula implementation."

Step 3 (Last resort — very rare, use with caution): Simplify or skip the function entirely.
   - If no non-banned replacement exists AND the function cannot be implemented via equations, consider simplifying the task requirements or omitting that specific function.
   - WARNING: This should be extremely rare. Skipping a function severely impacts modeling quality. Exhaust Step 1 and Step 2 thoroughly before resorting to this.
   - When this step is used, clearly document the omission and reason in selection_rationale.

------------------------------------------------------------
Ⅻ. 【[KEY★★★] Stop Condition】
- No limit on the maximum number of tool calls; you may call tools as many times as needed to fulfill the prompt requirements
- No strict time requirement for JSON output; output the JSON result once you have completed all prompt requirements (self-judge when ready)
- If search tools return no results, use get_tmf_spec directly with common component IDs
"""

    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the plan and select components.

        Args:
            state: Current state containing system_analysis and component_plan.

        Returns:
            State updates with selected components.
        """
        system_analysis = state.get("system_analysis", "")
        component_plan = state.get("component_plan", [])
        task_description = state.get("task_description", "")

        if not system_analysis and not component_plan and not task_description:
            logger.error("No system analysis, component plan, or task description provides")
            return {
                "components": [],
                "status": "failed",
            }

        self.log_action("Component selection started", f"Planned component categories: {len(component_plan)}")

        repair_round = self.is_repair_round(state)
        repair_context = self.build_repair_context(state) if repair_round else {}

        # Detect if Weather+Building bundle is needed from task description
        has_weather_building_bundle = self._detect_weather_building_bundle(task_description)

        # Create task message
        repair_context_text = self.build_repair_context_text(repair_context) if repair_round else ""
        prompt_override = self.wrap_prompt_with_repair_instructions(self.system_prompt) if repair_round else None

        # Pre-format state fields as standard JSON code blocks so that
        # `system_analysis` (raw LLM text from planner) and `component_plan`
        # (structured list-of-dicts) appear in a unified, model-friendly format
        # inside the human prompt below. Falls back to the original value when
        # parsing fails so no upstream information is ever lost.
        system_analysis_block = self._format_state_field_as_json_block(system_analysis)
        component_plan_block = (
            self._format_state_field_as_json_block(component_plan)
            if component_plan
            else "No explicit component plan"
        )

        if repair_round:
            task_message = HumanMessage(content=f"""You are working in a repair round for TRNSYS component selection.

【【Quick Index — Repair Round — HumanMessage】】
1. 【Repair Brief】 — Previous validation result, errors, failure analysis, earlier chain updates, your editing baseline
2. 【Repair-Round Instructions】 — 6-step: review previous → decide if changes needed → targeted updates → preserve structure → ensure consistency → same JSON format
3. 【Repair Round - Maximize DCK Tool Usage】 — 2 repair tools: search_dck_snippet, get_component_special_fields
4. 【Original Task】 — Task description for context
5. 【System Analysis】 — System analysis from planner
6. 【Component Plan】 — Planned component categories

1. 【Repair Brief】
{repair_context_text}
------------------------------------------------------------
2. 【Repair-Round Instructions】
1. Review the previous validation result, your previous output, and any updates already made earlier in this repair chain.
2. Decide whether component selection changes are actually required in your part for this round.
3. If changes are required, make targeted updates to the previous output instead of regenerating the full selection from scratch.
4. If no changes are required, keep the previous output logically unchanged and preserve the same output structure.
5. If you update anything, ensure the revised component selection still matches the current task and system analysis.
6. Return the result in the same JSON structure as the first round.

------------------------------------------------------------
3. 【Repair Round - Maximize DCK Tool Usage】
During this repair round, to improve the success rate of component selection fixes, you should make full use of DCK-related tools to retrieve information about existing validated component configurations from the training set:

Available DCK tools for component selection repair:
- search_dck_snippet: Retrieve component usage patterns from training DCKs
- get_component_special_fields: Analyze special field statistics (LABELS, External files) from training DCKs

These DCK tools help you retrieve information about existing models from the training set, which are manually modeled and verified accurate models that provide important reference value for your repair work. If you can obtain accurate model information from the training set that is highly related to the reported error, it will significantly improve the success rate of your repair.

------------------------------------------------------------
4. 【Original Task】
{task_description}

------------------------------------------------------------
5. 【System Analysis】
{system_analysis_block}

------------------------------------------------------------
6. 【Component Plan】
{component_plan_block}

Please review the repair context first, decide whether changes are needed, and then produce the updated result.
""")
        else:
            task_message = HumanMessage(content=f"""1. 【Task Instruction】
Please select TRNSYS components based on the following system planning result:

【【Quick Index — First Round — HumanMessage】】
1. 【Task Instruction】 — Instruction to select TRNSYS components based on system planning result
2. 【Task Description】 — Task description
3. 【System Analysis】 — System analysis from planner
4. 【Component Plan】 — Planned component categories
5. 【Requirements】 — 4 selection requirements: select Type, verify TMF, assign UNIT numbers, output JSON

------------------------------------------------------------
2. 【Task Description】
{task_description}

------------------------------------------------------------
3. 【System Analysis】
{system_analysis_block}

------------------------------------------------------------
4. 【Component Plan】
{component_plan_block}

------------------------------------------------------------
5. 【Requirements】
1. Select an appropriate TRNSYS Type component for each planned component category
2. Use the get_tmf_spec tool to verify that each selected component exists
3. Assign unique UNIT numbers starting from 2
4. Output the selection result in JSON format

Please begin component selection.
""")

        try:
            # Invoke agent
            response = self.invoke_agent_with_prompt([task_message], prompt_override=prompt_override)

            # Extract result
            result_text = self.extract_result(response)

            # Parse components from result
            components, parsed_data, json_str = self._parse_components(result_text)

            # FALLBACK: If no components but Weather+Building bundle detected
            if not components and has_weather_building_bundle:
                logger.warning("LLM did not generate components, using fallback for Weather+Building bundle")
                components = self._generate_fallback_components(task_description)

            self.log_action("Component selection completed", f"Selected {len(components)} components")

            # CRITICAL: Final check to ensure unit_number is first before returning
            components = self._ensure_unit_number_first(components)

            result = {
                "components": components,
                "status": "components_selected",
                "current_agent": "component_selector",
                "next_agent": "equation_generator",
            }
            if repair_round:
                result.update(
                    self.build_agent_output_snapshot_updates(
                        state,
                        result_text,
                        json_str=json_str,
                    )
                )
            else:
                self.cache_first_round_output(
                    state,
                    result_text,
                    json_str=json_str,
                )

            if repair_round:
                # Extract repair info from LLM output
                repair_summary, repair_detail = self.extract_repair_info_from_output(
                    parsed_data,
                    fallback_summary="Component selection updated for repair round" if components else "No component changes required",
                    fallback_detail="Adjusted component selection using validator feedback and prior chain notes" if components else "Reuse previous-round component output without changes",
                )
                repair_note = self.build_repair_note(
                    summary=repair_summary,
                    detail=repair_detail,
                    repair_type=self.REPAIR_TYPE_DIRECT,
                    changed=bool(components),
                    state=state,
                )
                repair_updates = self.build_repair_round_state_updates(
                    state=state,
                    raw_output=result_text,
                    repair_note=repair_note,
                    json_str=json_str,
                )
                result.update(repair_updates)

            return result

        except Exception as e:
            logger.error(f"Component selector error: {e}")
            # FALLBACK: Even on error, generate components if bundle detected
            if has_weather_building_bundle:
                logger.warning("Using fallback components due to error")
                fallback_components = self._generate_fallback_components(task_description)
                # CRITICAL: Ensure unit_number is first before returning
                fallback_components = self._ensure_unit_number_first(fallback_components)
                fallback_result = {
                    "components": fallback_components,
                    "status": "components_selected",
                    "current_agent": "component_selector",
                    "next_agent": "equation_generator",
                }
                if repair_round:
                    fallback_result.update(self.build_agent_output_snapshot_updates(state, ""))
                    repair_note = self.build_repair_note(
                        summary="Fallback component selection applied in repair round",
                        detail="Weather-building bundle fallback used due to expert invocation error",
                        repair_type=self.REPAIR_TYPE_DIRECT,
                        changed=True,
                        state=state,
                    )
                    fallback_result.update(
                        self.build_repair_round_state_updates(
                            state=state,
                            raw_output="",
                            repair_note=repair_note,
                            json_str="",
                        )
                    )
                else:
                    self.cache_first_round_output(state, "", json_str="")
                return fallback_result
            return {
                "components": [],
                "status": "failed",
            }

    def _detect_weather_building_bundle(self, task_description: str) -> bool:
        """
        Detect if the task requires Weather + Building bundle.

        Bundle is triggered ONLY when BOTH conditions are met:
        1. Task mentions building-related keywords
        2. Task explicitly provides a .b18/.b17/.bui building description file path

        If building is mentioned but no building file is provided, Type56 cannot
        function and the bundle must NOT be triggered.

        Args:
            task_description: Task description text.

        Returns:
            True if Weather+Building bundle is needed.
        """
        task_lower = task_description.lower()

        # Check for building-related keywords (excluding file extensions)
        has_building = any(kw in task_lower for kw in [
            "building", "structure", "type56", "thermal zone", "zone model",
            "office building", "office", "multi-zone", "multiple zones"
        ])

        # Check for explicit building description file path (.b18, .b17, .bui)
        has_building_file = bool(re.search(r'\S+\.b18\b', task_description, re.IGNORECASE)) or \
                            bool(re.search(r'\S+\.b17\b', task_description, re.IGNORECASE)) or \
                            bool(re.search(r'\S+\.bui\b', task_description, re.IGNORECASE))

        bundle_detected = has_building and has_building_file
        if bundle_detected:
            logger.info("Building bundle detected: building keywords AND building file (.b18/.b17/.bui) found")
        elif has_building and not has_building_file:
            logger.info("Building mentioned but NO .b18/.b17/.bui file provided - bundle NOT triggered, Type56 excluded")

        return bundle_detected

    def _generate_fallback_components(self, task_description: str) -> List[Dict[str, Any]]:
        """
        Generate fallback components for Weather + Building bundle.

        Args:
            task_description: Task description for context.

        Returns:
            List of default component dictionaries.
        """
        logger.info("Generating fallback components for Weather+Building bundle")

        # Determine weather file type from task description
        weather_type = "Type15-6"  # Default for .tm2 files
        if ".epw" in task_description.lower():
            weather_type = "Type15-3"
        elif ".tmy3" in task_description.lower():
            weather_type = "Type15-TMY3"
        elif ".tmy2" in task_description.lower():
            weather_type = "Type15-2"

        components = [
            {
                "unit_number": 2,
                "type_number": weather_type,
                "unit_name": "Weather_Data",
                "model_path": f".\\\\Weather\\\\{weather_type}.tmf",
                "purpose": "Weather data processing",
                "category": "weather"
            },
            {
                "unit_number": 3,
                "type_number": "Type56",
                "unit_name": "Building",
                "model_path": ".\\Loads and Structures\\Multi-Zone Building\\Type56.tmf",
                "purpose": "Multi-zone building thermal simulation",
                "category": "building"
            },
            {
                "unit_number": 4,
                "type_number": "Type77",
                "unit_name": "Ground_Temperature",
                "model_path": ".\\Physical Phenomena\\Undisturbed Ground Temperature Profile\\Type77.tmf",
                "purpose": "Ground temperature boundary conditions",
                "category": "ground"
            },
            {
                "unit_number": 5,
                "type_number": "Type65d",
                "unit_name": "Online_Plotter",
                "model_path": ".\\Output\\Online Plotter\\Online Plotter Without File\\Type65d.tmf",
                "purpose": "Real-time data visualization",
                "category": "output"
            },
            {
                "unit_number": 6,
                "type_number": "Type25a",
                "unit_name": "Data_Printer",
                "model_path": ".\\Output\\Printer\\Unformatted\\Type25a.tmf",
                "purpose": "Data output to file",
                "category": "output"
            }
        ]

        return components

    def _format_state_field_as_json_block(self, value: Any) -> str:
        """
        Render a state field as a unified, model-friendly ```json ... ``` block.

        Goal: make ``system_analysis`` (raw LLM text from the planner) and
        ``component_plan`` (structured list-of-dicts) share an identical visual
        and structural form inside the human prompt, so the LLM does not need
        to switch between two different presentation styles for upstream data.

        Behavior:
        - dict / list: serialize with ``json.dumps(indent=2, ensure_ascii=False)``
          and wrap in a ```json fenced code block.
        - str: best-effort parse the string as JSON (either embedded inside an
          existing ```json fence or as a whole), then re-serialize in the same
          unified form. If parsing fails at every step, fall back to the
          ORIGINAL string untouched so no upstream information is ever lost.
        - other types: convert via ``str()`` and return unfenced (extremely
          unlikely path for this agent, kept only as a safety net).

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

        # Raw string path (typical for ``system_analysis`` from planner)
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
            # lose information that the planner produced.
            return value

        # Safety net for unexpected types
        return str(value)

    def _parse_components(self, result_text: str) -> tuple[List[Dict[str, Any]], Dict[str, Any], str]:
        """
        Parse the component selection result.

        Args:
            result_text: Raw result text from LLM.

        Returns:
            Tuple of (components list, full parsed data dict including repair info, extracted JSON string)
        """
        try:
            # Look for JSON block
            json_match = re.search(r'```json\s*(.*?)\s*```', result_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1).strip()
                data = json.loads(json_str)
            else:
                # Try to parse the entire text as JSON
                json_str = result_text.strip()
                data = json.loads(json_str)

            components = data.get("components", [])

            # CRITICAL: Ensure unit_number is first field immediately after JSON parsing
            components = self._ensure_unit_number_first(components)

            # Validate and fix UNIT number uniqueness
            components = self._ensure_unique_unit_numbers(components)

            return components, data, json_str

        except json.JSONDecodeError:
            logger.warning("Could not parse JSON from component selector result")
            return [], {}, ""

    def _ensure_unique_unit_numbers(self, components: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Ensure all components have unique UNIT numbers.

        IMPORTANT: Rebuild component dict with unit_number first to maintain field order.

        Args:
            components: List of component dictionaries.

        Returns:
            Components with unique UNIT numbers and unit_number as first field.
        """
        seen_units = set()
        next_unit = 2
        result = []

        for comp in components:
            unit_num = comp.get("unit_number")

            # If unit number is missing or duplicate, assign a new one
            if unit_num is None or unit_num in seen_units:
                while next_unit in seen_units:
                    next_unit += 1
                unit_num = next_unit
                logger.warning(f"Reassigned UNIT number to {next_unit} for component {comp.get('type_number', 'unknown')}")

            # Rebuild component dict with unit_number first
            new_comp = {"unit_number": unit_num}

            # Copy all other fields from original component
            for key, value in comp.items():
                if key != "unit_number":
                    new_comp[key] = value

            seen_units.add(unit_num)
            next_unit = max(next_unit, unit_num + 1)
            result.append(new_comp)

        return result

    def select_components(
        self,
        task_description: str,
        system_analysis: str,
        component_plan: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Standalone method to select components.

        Args:
            task_description: Original task description.
            system_analysis: System analysis from planner.
            component_plan: Component plan from planner.

        Returns:
            List of selected components.
        """
        state = {
            "task_description": task_description,
            "system_analysis": system_analysis,
            "component_plan": component_plan,
        }
        result = self.process(state)
        return result.get("components", [])
