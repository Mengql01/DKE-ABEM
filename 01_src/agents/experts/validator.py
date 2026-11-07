"""
Validator Expert Agent for TRNSYS DCK Generator.

Responsible for assembling DCK files and validating them using TrnEXE64.exe.
This is the final expert in the workflow and handles iterative error correction.

Key Responsibilities:
1. Assemble complete DCK files from component data (using DCK module)
2. Validate DCK files using TrnEXE64.exe (using DCK module)
3. Analyze validation errors using LLM (NOT hardcoded rules)
4. Record success/failure experiences to Memory
5. Provide repair suggestions for failed validations
6. Compare with training DCK snippets and fix missing special fields

Design Principles:
- DCK assembly uses the dedicated DCK module (templates + format checker)
- Error analysis uses LLM, NOT hardcoded if-else rules
- Supports iterative repair workflow
- Records experiences for future learning
- Uses training data comparison to fix missing fields (LABELS, External files, etc.)

Integration with DCK Module (Phase 5):
- Uses DCKAssembler for DCK file assembly
- Uses DCKValidator for TrnEXE64 validation
- Uses DCKFormatChecker for pre-validation
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from .base_expert import BaseExpert
from ...core.llm_client import call_validator_with_search

# Import DCK module components
try:
    from ...dck import (
        DCKAssembler,
        DCKValidator as DCKValidatorModule,
        DCKFormatChecker,
        AssemblyResult,
        DCKValidationResult,
        get_assembler,
        get_validator as get_dck_validator,
    )
    DCK_MODULE_AVAILABLE = True
except ImportError:
    DCK_MODULE_AVAILABLE = False

# Import DCK snippet tools for training data comparison
try:
    from ..tools.dck_snippet_tools import (
        get_special_fields_for_type,
    )
    DCK_SNIPPET_TOOLS_AVAILABLE = True
except ImportError:
    DCK_SNIPPET_TOOLS_AVAILABLE = False

logger = logging.getLogger(__name__)


class ValidatorExpert(BaseExpert):
    """
    Validator Expert Agent.

    Assembles and validates TRNSYS DCK files.

    Now integrated with DCK module (Phase 5) for:
    - Template-based DCK assembly
    - Format pre-validation
    - TrnEXE64.exe execution
    - LLM-driven error analysis
    """

    def __init__(self, *args, **kwargs):
        """Initialize with DCK module components."""
        super().__init__(*args, **kwargs)

        # Initialize DCK module components if available
        self._dck_assembler = None
        self._dck_validator = None

        if DCK_MODULE_AVAILABLE:
            try:
                self._dck_assembler = get_assembler(method=self.method)
                self._dck_validator = get_dck_validator()
                logger.info("DCK module components initialized successfully")
            except Exception as e:
                logger.warning(f"Could not initialize DCK module: {e}")

    @property
    def dck_assembler(self) -> Optional[DCKAssembler]:
        """Get DCK assembler instance."""
        if self._dck_assembler is None and DCK_MODULE_AVAILABLE:
            try:
                self._dck_assembler = get_assembler(method=self.method)
            except Exception as e:
                logger.warning(f"Could not get DCK assembler: {e}")
        return self._dck_assembler

    @property
    def dck_validator(self) -> Optional[DCKValidatorModule]:
        """Get DCK validator instance."""
        if self._dck_validator is None and DCK_MODULE_AVAILABLE:
            try:
                self._dck_validator = get_dck_validator()
            except Exception as e:
                logger.warning(f"Could not get DCK validator: {e}")
        return self._dck_validator

    @property
    def name(self) -> str:
        return "validator"

    @property
    def system_prompt(self) -> str:
        return """Ⅰ. 【[CORE★★★★★] Role Definition】
You are a TRNSYS validation expert. Your task is to assemble the DCK file and validate its correctness.

【【Quick Index — First Round — SystemMessage】】
Ⅰ. 【[CORE★★★★★] Role Definition】 — TRNSYS validation expert, assemble DCK and validate correctness
Ⅱ. 【[CORE★★★★★] Workflow - FOLLOW STEP BY STEP】 — 6-step: retrieve format → assemble DCK → validate → parse errors → analyze repair → record experience
Ⅲ. 【[CORE★★★★★] Output Format - ALL IN ENGLISH】 — JSON: root_cause + responsible_agents + fix_suggestions + priority; error analysis must be valid JSON only
Ⅳ. 【[CORE★★★★★] Maximize Tool Usage】 — 15 tools across 4 categories for validation and error analysis
Ⅴ. 【[CORE★★★★★] Repair Round - Maximize DCK Tool Usage】 — DCK snippet tools for repair: search_dck_snippet, get_component_special_fields
Ⅵ. 【[KEY★★★] Expert Agents Available for Repair Routing】 — 5 repair experts: component_selector, equation_generator, parameter_configurator, connection_generator, layout_designer
Ⅶ. 【[KEY★★★] Error Types and Responsible Experts】 — Error-to-expert mapping: component/equation/parameter/connection/layout errors
Ⅷ. 【[KEY★★★] Responsibilities】 — 4 tasks: assemble DCK, validate with TrnEXE64, analyze errors, record experiences
Ⅸ. 【[KEY★★★] DCK File Structure Example】 — Complete DCK example with VERSION, Control cards, UNIT/TYPE, PARAMETERS, INPUTS, EQUATIONS, LABELS, LINK_STYLE
Ⅹ. 【[KEY★★★] Important Principles】 — Learn from examples, LLM-based analysis, record experience, parameter/input counts match TMF

------------------------------------------------------------
Ⅱ. 【[CORE★★★★★] Workflow - FOLLOW STEP BY STEP】
1. Use tools to retrieve the format of similar DCK files as references (e.g., search_dck_snippet, rag_search_dck_format_examples)
2. Assemble the DCK file from the component data
3. Validation is executed automatically using TrnEXE64.exe via DCK module
4. If validation fails, analyze the provided error information
5. Determine the repair strategy based on error analysis
6. Record experience in Memory

------------------------------------------------------------
Ⅲ. 【[CORE★★★★★] Output Format - ALL IN ENGLISH】
When analyzing validation errors, output ONLY valid JSON format. Do NOT output any explanations, descriptions, or text before or after the JSON.
Do NOT include markdown code blocks or any other text.

Output ONLY this JSON structure:
```json
{{
    "root_cause": "Root cause of the error",
    "responsible_agents": ["agent_name"],
    "fix_suggestions": ["Repair suggestion 1", "Repair suggestion 2"],
    "priority": "high/medium/low"
}}
```

**JSON Structure Explanation:**
- **root_cause**: String, concise description of the root cause (required)
  - Should identify the specific TRNSYS error type and affected component
  - Example: "Parameter count mismatch for Type15-6: expected 9 but found 8"
- **responsible_agents**: Array of strings, agent names that should handle repair (required, at least 1)
  - Valid agent names: "component_selector", "equation_generator", "parameter_configurator", "connection_generator", "layout_designer"
  - Do NOT include "planner" or "validator"
  - Multiple agents can be listed when errors span multiple domains
- **fix_suggestions**: Array of strings, specific actionable repair suggestions (required, at least 1)
  - Each suggestion should be concrete and actionable, not vague
  - Example: "Add missing parameter 9 (Azimuth of surface) to Type15-6 with default value 0"
- **priority**: String, one of "high", "medium", or "low" (required)
  - "high": Errors that prevent simulation from running (fatal errors)
  - "medium": Errors that cause incorrect results (warning-level)
  - "low": Minor formatting or style issues
- ALL output must be in ENGLISH - no Chinese characters

------------------------------------------------------------
Ⅳ. 【[CORE★★★★★] Maximize Tool Usage】
You should make full use of the tools available to you to gather sufficient relevant information.
You have 15 tools available across 4 categories:

**1. RAG Tools (7 tools) - Search training data for examples:**
- rag_search_component_specs: Search component specifications from training data
- rag_search_dck_format_examples: Search DCK format examples
- rag_search_connection_examples: Search component connection examples
- rag_search_equation_examples: Search EQUATIONS block examples
- rag_search_parameter_examples: Search parameter configuration examples
- rag_search_input_examples: Search input configuration examples
- rag_search_global_for_dck_and_tmf: Comprehensive search (system + component + format)

**2. TMF Tools (3 tools) - Verify component specifications:**
- get_tmf_spec: Get complete TMF specification for a component type
- validate_parameter_count: Verify if parameter count matches TMF specification
- validate_input_count: Verify if input count matches TMF specification

**3. Equation Tools (2 tools) - Verify equation syntax:**
- validate_equation_syntax: Validate equation syntax
- validate_equations_block: Validate entire EQUATIONS block

**4. DCK Snippet Tools (3 tools) - Search training DCK snippets:**
- search_dck_snippet: Search complete DCK snippet for specific component type
- get_component_special_fields: Analyze special fields (LABELS, External files, etc.)
- rank_components_by_usage: Rank components by usage frequency in training set

**5. Connection Pattern Tools (3 tools) - Search connection patterns:**
- search_connection_patterns: Search for connection patterns between component types
- get_connection_pattern: Get detailed connection pattern for specific component pair
- find_dcks_with_components: Find training DCK files containing specific components

These tools help you retrieve information about existing validated models from the training set,
which are manually modeled and verified accurate models that provide important reference value for your work.

------------------------------------------------------------
Ⅴ. 【[KEY★★★] Repair Round - Maximize DCK Tool Usage】
When working in a repair round to fix validation errors, you should make full use of DCK-related tools to retrieve information about existing validated models from the training set that are highly related to the reported errors:

**Available DCK tools for validation repair:**
- search_dck_snippet: Search complete DCK snippet for specific component type from training set
- get_component_special_fields: Analyze special fields (LABELS, External files, etc.) from training DCK snippets
- rank_components_by_usage: Rank components by usage frequency to understand common patterns
- rag_search_dck_format_examples: Search DCK format examples from training data
- rag_search_global_for_dck_and_tmf: Comprehensive search across system, component, and format information

These DCK tools help you retrieve information about existing validated models from the training set, which are manually modeled and verified accurate models that provide important reference value for your repair work. If you can obtain accurate model information from the training set that is highly related to the reported validation error, it will significantly improve the success rate of your repair.

------------------------------------------------------------
Ⅵ. 【[KEY★★★] Expert Agents Available for Repair Routing】
When analyzing validation errors, you need to identify which expert agents should be responsible for fixing the errors. The available repair experts are (excluding planner and validator):

1. component_selector - Select appropriate TRNSYS components from 765-component TMF library based on functional requirements, assign unique UNIT numbers, verify component existence and variants using TMF tools

2. equation_generator - Generate EQUATIONS blocks (special components) for calculations, data post-processing, control logic, signal simulation, and component interface bridging using TRNSYS equation syntax

3. parameter_configurator - Configure PARAMETERS blocks (parameter counts must match TMF specs), EXTERNAL_FILES (weather/building files with unique logical units), and SPECIFIC_PROPERTIES for each component

4. connection_generator - Generate INPUTS blocks for all components and equations, create connections using [unit, output] format, handle unconnected inputs with 0,0, ensure input counts match TMF specifications

5. layout_designer - Design component layout with LEFT-TO-RIGHT orientation (horizontal span > vertical span), assign position coordinates based on data flow topology, generate LINK_STYLE blocks with CONNECTION_SET format

------------------------------------------------------------
Ⅶ. 【[KEY★★★] Error Types and Responsible Experts】
When determining responsible_agents in your error analysis, consider:
- Component selection errors (wrong component type, missing component) → component_selector
- Equation syntax errors, calculation errors, control logic errors → equation_generator
- Parameter count mismatch, parameter value errors, external file path errors, logical unit conflicts → parameter_configurator
- Input count mismatch, connection errors, invalid [unit,output] references → connection_generator
- Layout errors, unit number conflicts, position coordinate errors, LINK_STYLE format errors → layout_designer
- Multiple error types → select multiple experts for repair

IMPORTANT: Do not include "planner" or "validator" in the responsible_agents list. Planner is first-round only, and validator has just reported the error.

------------------------------------------------------------
Ⅷ. 【[KEY★★★] Responsibilities】
1. Assemble a complete DCK file based on component, parameter, and connection information
2. Call TrnEXE64.exe to validate the DCK file
3. Analyze validation errors and provide repair suggestions
4. Record successful and failed experiences

------------------------------------------------------------
Ⅸ. 【[KEY★★★] DCK File Structure Example】
```
VERSION 18
*******************************************************************************
*** TRNSYS input file (deck) generated by Large Language Model
*** on Sunday, December 14, 2025 at 12:43
***
*******************************************************************************


*******************************************************************************
*** Units
*******************************************************************************

*******************************************************************************
*** Control cards
*******************************************************************************
* START, STOP and STEP
CONSTANTS 3
START=4344
STOP=5088
STEP=0.125
SIMULATION 	 START	 STOP	 STEP	! Start time	End time	Time step
TOLERANCES 0.001 0.001			! Integration	 Convergence
LIMITS 50 50 50				! Max iterations	Max warnings	Trace limit
DFQ 1					! TRNSYS numerical integration solver method
WIDTH 120				! TRNSYS output file width, number of characters
LIST 					! NOLIST statement
					! MAP statement
SOLVER 0 1 1				! Solver statement	Minimum relaxation factor	Maximum relaxation factor
NAN_CHECK 0				! Nan DEBUG statement
OVERWRITE_CHECK 0			! Overwrite DEBUG statement
TIME_REPORT 0			! disable time report
EQSOLVER 0				! EQUATION SOLVER statement
* User defined CONSTANTS
*$USER_CONSTANTS

*$USER_CONSTANTS_END


* Model "TMY-3 Data Processor" (Type 15)
*

UNIT 3 TYPE 15	 TMY-3 Data Processor
*$UNIT_NAME TMY-3 Data Processor
*$MODEL .\.\Weather Data Reading and Processing\Standard Format\TMY3\Type15-TMY3.tmf
*$POSITION 104 238
*$LAYER Main #
PARAMETERS 9
7		! 1 File Type
36		! 2 Logical unit
3		! 3 Tilted Surface Radiation Mode
0.2		! 4 Ground reflectance: no albedo reported
0.7		! 5 Not used
1		! 6 Number of surfaces
1		! 7 Tracking mode
0		! 8 Slope of surface
0		! 9 Azimuth of surface
*** External files
ASSIGN "D:\\TRNSYS18_03\\Weather\\US-TMY3\\US_WI_MADISON_DANE_CO_REGIONAL_ARPT_[ISIS]_726410.tmy3" 36
*|? Which file contains the TMY-2 weather data? |1000
*------------------------------------------------------------------------------

* Model "Online Plotter" (Type 65)
*

UNIT 5 TYPE 65	 Online Plotter
*$UNIT_NAME Online Plotter
*$MODEL .\.\Output\Online Plotter\Online Plotter Without File\Type65d.tmf
*$POSITION 765 308
*$LAYER Main #
PARAMETERS 12
7		! 1 Nb. of left-axis variables
1		! 2 Nb. of right-axis variables
-20		! 3 Left axis minimum
50		! 4 Left axis maximum
0		! 5 Right axis minimum
5		! 6 Right axis maximum
4		! 7 Number of plots per simulation
7		! 8 X-axis gridpoints
0		! 9 Shut off Online w/o removing
-1		! 10 Logical unit for output file
0		! 11 Output file units
0		! 12 Output file delimiter
INPUTS 8
3,1 		! TMY-3 Data Processor:Dry bulb temperature ->Left axis variable-1
3,2 		! TMY-3 Data Processor:Dew point temperature ->Left axis variable-2
T_in		! Inlet Temp:T_in ->Left axis variable-3
7,1 		! Type666:Chilled Water Temperature ->Left axis variable-4
7,3 		! Type666:Cooling Water Temperature ->Left axis variable-5
10,1 		! Pipe for capacitance:Outlet temperature ->Left axis variable-6
8,1 		! Type510:Fluid Outlet Temperature ->Left axis variable-7
7,8 		! Type666:C.O.P. ->Right axis variable
*** INITIAL INPUT VALUES
DryBulb DewPoint CHW_in CHW_out CW_out Pipe CW_in_OR_Tower_out COP
LABELS  3
"Temperatures"
"Not used"
"Graph 1"
*------------------------------------------------------------------------------

* EQUATIONS "Inlet Temp"
*
EQUATIONS 1
T_in = 20+3*sin(time*10)
*$UNIT_NAME Inlet Temp
*$LAYER Main
*$POSITION 295 61
*$UNIT_NUMBER 6

*------------------------------------------------------------------------------


* Model "Type666" (Type 666)
*

UNIT 7 TYPE 666	 Type666
*$UNIT_NAME Type666
*$MODEL .\.\HVAC Library (TESS)\Chillers\Water-Cooled Chiller\Type666.tmf
*$POSITION 382 134
*$LAYER Main #
*$# Water-Cooled Chiller
PARAMETERS 9
1266072.153961		! 1 Rated Capacity
3		! 2 Rated C.O.P.
37		! 3 Logical Unit - Performance Data
38		! 4 Logical Unit - PLR Data
4.19		! 5 CHW Fluid Specific Heat
4.19		! 6 CW Fluid Specific Heat
6		! 7 Number of CW Points
6		! 8 Number of CHW Points
5		! 9 Number of PLRs
INPUTS 6
T_in		! Inlet Temp:T_in ->Chilled Water Inlet Temperature
0,0		! [unconnected] Chilled Water Flowrate
8,1 		! Type510:Fluid Outlet Temperature ->Cooling Water Temperature
8,2 		! Type510:Outlet Fluid Flowrate ->Cooling Water Flowrate
0,0		! [unconnected] CHW Set Point Temperature
0,0		! [unconnected] Chiller Control Signal
*** INITIAL INPUT VALUES
12.22 10000 30 20000 7 1
*** External files
ASSIGN "D:\\TRNSYS18_03\\Tess Models\\SampleCatalogData\\WaterCooledChiller\\Samp_C.Dat" 37
*|? Which file contains the chiller performance data? |1000
ASSIGN "D:\\TRNSYS18_03\\Tess Models\\SampleCatalogData\\WaterCooledChiller\\Samp_PLR.Dat" 38
*|? Which file contains the part-load performance data? |1000
*------------------------------------------------------------------------------

* Model "Type510" (Type 510)
*

UNIT 8 TYPE 510	 Type510
*$UNIT_NAME Type510
*$MODEL .\.\HVAC Library (TESS)\Closed Circuit Cooling Tower\Type510.tmf
*$POSITION 256 238
*$LAYER Main #
*$# Closed Circuit Cooling Tower
PARAMETERS 15
2		! 1 Humidity Mode
48.888911		! 2 Design Inlet Fluid Temperature
40.555577		! 3 Design Outlet Fluid Temperature
40000		! 4 Design Fluid Flowrate
4.19		! 5 Fluid Specific Heat
35.000022		! 6 Design Ambient Air Temperature
25.555577		! 7 Design Wet Bulb Temperature
100000		! 8 Design Air Flowrate
1		! 9 Air Pressure at Design Conditions
13422.818772		! 10 Rated Fan Power
4		! 11 Number of Power Coefficients
0		! 12 Power Coefficient-1
0		! 13 Power Coefficient-2
0		! 14 Power Coefficient-3
1		! 15 Power Coefficient-4
INPUTS 8
10,1 		! Pipe for capacitance:Outlet temperature ->Fluid Inlet Temperature
10,2 		! Pipe for capacitance:Outlet flow rate ->Fluid Flowrate
3,1 		! TMY-3 Data Processor:Dry bulb temperature ->Ambient Temperature
3,6 		! TMY-3 Data Processor:Humidity ratio ->Ambient Humidity Ratio
3,7 		! TMY-3 Data Processor:Percent relative humidity ->Ambient % Relative Humidity
3,10 		! TMY-3 Data Processor:Atmospheric pressure ->Ambient Air Pressure
0,0		! [unconnected] Fan Control Signal
0,0		! [unconnected] Desired Outlet Temperature
*** INITIAL INPUT VALUES
10 20000 20 0.005 50 1 1 30
*------------------------------------------------------------------------------

* Model "Type3d" (Type 3)
*

UNIT 9 TYPE 3	 Type3d
*$UNIT_NAME Type3d
*$MODEL .\.\Obsolete\Pumps\Single Speed\Type3d.tmf
*$POSITION 474 135
*$LAYER Main #
PARAMETERS 4
40000		! 1 Maximum flow rate
4.19		! 2 Fluid specific heat
2684.563754		! 3 Maximum power
0		! 4 Conversion coefficient
INPUTS 3
7,3 		! Type666:Cooling Water Temperature ->Inlet fluid temperature
7,4 		! Type666:Cooling Water Flowrate ->Inlet mass flow rate
0,0		! [unconnected] Control signal
*** INITIAL INPUT VALUES
20 100 1
*------------------------------------------------------------------------------

* Model "Pipe for capacitance" (Type 31)
*

UNIT 10 TYPE 31	 Pipe for capacitance
*$UNIT_NAME Pipe for capacitance
*$MODEL .\.\Hydronics\Pipe\Type31.tmf
*$POSITION 586 135
*$LAYER Water Loop #
PARAMETERS 6
0.2		! 1 Inside diameter
160		! 2 Pipe length
0		! 3 Loss coefficient
1000		! 4 Fluid density
4.19		! 5 Fluid specific heat
10		! 6 Initial fluid temperature
INPUTS 3
9,1 		! Type3d:Outlet fluid temperature ->Inlet temperature
9,2 		! Type3d:Outlet flow rate ->Inlet flow rate
0,0		! [unconnected] Environment temperature
*** INITIAL INPUT VALUES
10 0 10
*------------------------------------------------------------------------------

END
*!LINK_STYLE
*!LINK 3:5
*!CONNECTION_SET 20:40:0:20:2,1:10789024:1:0:0:502,346:502,396:1081,396:1081,396:1143,396
*!LINK 8:5
*!CONNECTION_SET 20:40:0:20:7:10789024:1:0:0:648,349:648,399:1075,399:1075,399:1137,399
*!LINK 10:5
*!CONNECTION_SET 40:20:0:20:6:10789024:1:0:0:1003,228:1109,228:1109,401:1142,401
*!LINK 7:5
*!CONNECTION_SET 40:40:0:20:8,5,4:16776960:0:4:0:795,205:815,205:815,360:1138,360
*!LINK 6:5
*!CONNECTION_SET 40:20:0:20:3:10789024:1:0:0:710,142:1078,142:1078,389:1140,389
*!LINK 3:8
*!CONNECTION_SET 40:20:0:20:6,5,4,3:65535:0:4:0:510,332:603,332:622,332
*!LINK 8:7
*!CONNECTION_SET 20:0:0:19:4,3:16711935:0:4:0:631,318:631,312:718,312:718,235:739,235
*!LINK 10:8
*!CONNECTION_SET 40:20:40:40:2,1:16711935:0:4:0:1003,222:1053,222:1053,345:673,345
*!LINK 7:9
*!CONNECTION_SET 40:19:0:20:2,1:16711935:0:4:0:792,230:822,230:849,230
*!LINK 9:10
*!CONNECTION_SET 40:20:0:20:2,1:16711935:0:4:0:888,228:905,228:955,228
*!LINK 6:7
*!CONNECTION_SET 40:20:20:0:1:16711680:0:4:1:607,-121:675,-121:675,-64
*!LINK_STYLE_END

```

------------------------------------------------------------
Ⅹ. 【[KEY★★★] Important Principles】
- Learn DCK formatting from retrieved examples instead of inventing a format yourself
- Use LLM-based analysis for error analysis rather than hard-coded rules
- Record successful experience after validation succeeds
- Provide specific repair suggestions after validation fails
- Parameter counts and input counts must match the TMF specifications
"""

    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process state, assemble DCK, and validate.

        Args:
            state: Current state containing all component data.

        Returns:
            State updates with validation results.
        """
        components = state.get("components", [])
        equations = state.get("equations", [])
        constants = state.get("constants", [])
        simulation_settings = state.get("simulation_settings", {})
        layout_styles = state.get("layout_styles", [])
        link_style_lines = state.get("link_style_lines", [])
        task_id = state.get("task_id", "unknown")
        task_description = state.get("task_description", "")

        if not components:
            logger.error("No components provided for validation")
            return {
                "validation_result": "failed",
                "validation_errors": [{"type": "no_components", "message": "No component data provided"}],
                "status": "failed",
            }

        self.log_action("Starting validation", f"Component count: {len(components)}")

        try:
            # Step 1: Assemble DCK
            dck_content = self._assemble_dck(
                components=components,
                equations=equations,
                constants=constants,
                simulation_settings=simulation_settings,
                layout_styles=layout_styles,
                link_style_lines=link_style_lines,
                task_description=task_description,
            )

            # Step 2: Validate using TrnEXE64
            validation_result = self._validate_dck(dck_content, task_id)

            if validation_result["success"]:
                # Success: record experience
                self._record_success(state, dck_content)

                self.log_action("Validation successful", "DCK file passed validation")

                next_validator_round = state.get("validator_round", 0) + 1
                snapshot = {
                    "validation_result": "success",
                    "validation_errors": [],
                    "error_analysis": {},
                    "validator_round": next_validator_round,
                    "current_round_repair_notes": state.get("current_round_repair_notes", []),
                }

                # V3.4: Store validation files externally to reduce checkpoint size
                file_version = validation_result.get("file_version", {})
                # Note: Store is not directly accessible from process() method
                # For now, keep using state storage (backward compatible)
                # TODO: Refactor to pass config/store through process() signature in future
                validation_file_update = {
                    "validation_file_versions": [file_version]
                }

                return {
                    "dck_content": dck_content,
                    "validation_result": "success",
                    "validation_errors": [],
                    **validation_file_update,
                    "last_validator_snapshot": snapshot,
                    "validator_round": next_validator_round,
                    "status": "completed",
                    "current_agent": "validator",
                    "next_agent": "FINISH",
                    "repair_execution_chain": [],
                    "repair_targets": [],
                    "human_expert_error_analysis": {},  # Clear on success
                }
            else:
                # Failure: analyze errors
                # Check if human expert has provided error analysis (bypasses LLM)
                human_ea = state.get("human_expert_error_analysis", {}) or {}
                has_human_analysis = any(
                    human_ea.get(k) for k in ("priority", "root_cause", "fix_suggestions", "responsible_agents")
                )

                if has_human_analysis:
                    # Human expert error analysis provided — skip LLM, use directly
                    error_analysis = {
                        "root_cause": human_ea.get("root_cause", "Human expert diagnosis"),
                        "responsible_agents": human_ea.get("responsible_agents", ["validator"]),
                        "fix_suggestions": human_ea.get("fix_suggestions", []),
                        "priority": human_ea.get("priority", "high"),
                    }
                    logger.info(
                        "Using human expert error analysis (LLM skipped): root_cause=%s, agents=%s",
                        error_analysis["root_cause"],
                        error_analysis["responsible_agents"],
                    )
                else:
                    # No human expert input — use LLM analysis as usual
                    error_analysis = self._analyze_errors(
                        validation_result["errors"],
                        dck_content,
                    )

                self.log_action("Validation failed", f"Found {len(validation_result['errors'])} errors")

                # Bound the size of validation_errors written to LangGraph state.
                # A pathological TRNSYS run can emit 70k+ near-duplicate errors;
                # since this list ends up replicated across every checkpoint blob,
                # leaving it untruncated bloats state to multi-GB. The *prompt*
                # path (_analyze_errors -> _build_analysis_prompt) still receives
                # the full list and applies its own dedup, so analysis quality is
                # unchanged.
                errors_for_state = self._dedupe_and_truncate_errors(
                    validation_result["errors"]
                )

                next_validator_round = state.get("validator_round", 0) + 1
                snapshot = {
                    "validation_result": "failed",
                    "validation_errors": errors_for_state,
                    "error_analysis": error_analysis,
                    "validator_round": next_validator_round,
                    "current_round_repair_notes": state.get("current_round_repair_notes", []),
                }

                # V3.4: Keep using state storage for now (Store integration deferred)
                file_version = validation_result.get("file_version", {})

                return {
                    "dck_content": dck_content,
                    "validation_result": "failed",
                    "validation_errors": errors_for_state,
                    "validation_file_versions": [file_version],
                    "error_analysis": error_analysis,
                    "last_validator_snapshot": snapshot,
                    "validator_round": next_validator_round,
                    "status": "validation_failed",
                    "current_agent": "validator",
                    "next_agent": "supervisor",  # Return to supervisor for repair routing
                    "repair_mode": True,
                    "repair_execution_chain": [],
                    "human_expert_error_analysis": {},  # Clear after use to avoid repeated application
                }

        except Exception as e:
            logger.error(f"Validator error: {e}")
            next_validator_round = state.get("validator_round", 0) + 1
            snapshot = {
                "validation_result": "error",
                "validation_errors": [{"type": "exception", "message": str(e)}],
                "error_analysis": {},
                "validator_round": next_validator_round,
                "current_round_repair_notes": state.get("current_round_repair_notes", []),
            }
            return {
                "validation_result": "error",
                "validation_errors": [{"type": "exception", "message": str(e)}],
                "last_validator_snapshot": snapshot,
                "validator_round": next_validator_round,
                "repair_mode": True,
                "status": "failed",
                "current_agent": "validator",
                "next_agent": "supervisor",
            }

    def _assemble_dck(
        self,
        components: List[Dict[str, Any]],
        equations: List[Dict[str, Any]],
        constants: List[Dict[str, Any]],
        simulation_settings: Dict[str, Any],
        layout_styles: List[Dict[str, Any]],
        link_style_lines: List[Any],
        task_description: str,
    ) -> str:
        """
        Assemble a complete DCK file using src/dck module.

        Args:
            components: List of component dictionaries.
            equations: List of equation block dictionaries.
            constants: List of constant dictionaries.
            simulation_settings: Simulation settings dictionary.
            layout_styles: List of layout style dictionaries.
            link_style_lines: Pre-generated LINK_STYLE lines from layout_designer.
            task_description: Task description for header.

        Returns:
            Assembled DCK content string.

        Raises:
            RuntimeError: If DCK module is not available or assembly fails.
        """
        if self.dck_assembler is None:
            raise RuntimeError("DCK assembler module is not available")

        # Fix missing special fields by comparing with training data
        # Pass task_description to extract file paths
        components = self._fix_missing_special_fields(components, task_description)

        result = self.dck_assembler.assemble(
            components=components,
            equations=equations,
            constants=constants,
            simulation_settings=simulation_settings,
            layout_styles=layout_styles,
            link_style_lines=link_style_lines,
            task_description=task_description,
            optimize_with_llm=False,
        )

        if result.dck_content:
            logger.info(f"DCK assembled using DCK module (errors: {result.error_count})")
            if result.format_errors:
                for error in result.format_errors[:5]:
                    logger.warning(f"Format issue: {error}")

            # Post-process: add DERIVATIVES blocks for thermal storage components
            dck_content = self._add_derivatives_for_storage_tanks(
                result.dck_content, components, task_description
            )
            return dck_content

        # Assembly failed with no content
        error_msgs = [str(e) for e in result.format_errors[:5]] if result.format_errors else ["Unknown assembly error"]
        raise RuntimeError(f"DCK assembly failed: {'; '.join(error_msgs)}")

    def _validate_dck(self, dck_content: str, task_id: str) -> Dict[str, Any]:
        """
        Validate DCK using TrnEXE64.exe via src/dck module.

        Only TrnEXE64 errors are collected (format_checker excluded).
        Errors use the TRNSYS standard format:
            {index, severity, time_h, unit_number, type_number, detail_message}

        Args:
            dck_content: DCK file content to validate.
            task_id: Task identifier for temp file naming.

        Returns:
            Dictionary with success status, errors in new format, and file version info.

        Raises:
            RuntimeError: If DCK validator module is not available.
        """
        if self.dck_validator is None:
            raise RuntimeError("DCK validator module is not available")

        result = self.dck_validator.validate(
            dck_content=dck_content,
            task_id=task_id,
            analyze_errors=True,
        )

        # Convert TRNSYSError to new standard format (no format_checker errors)
        errors = []
        for te in result.trnsys_errors:
            errors.append({
                "index": getattr(te, '_index', len(errors) + 1),
                "severity": getattr(te, 'error_type', te.severity),
                "time_h": getattr(te, '_time_h', "0.000000"),
                "unit_number": getattr(te, '_raw_unit', None) or te.unit_number,
                "type_number": getattr(te, '_raw_type', None) or getattr(te, '_type_number', None),
                "detail_message": te.message,
                "reported_information": getattr(te, '_reported_info', ""),
            })

        logger.info(f"DCK validated using DCK module (success: {result.success}, errors: {len(errors)})")

        # Build file version record for state.
        # NOTE: log_content / lst_content / dck_content are intentionally filtered here.
        # A single TRNSYS run can produce 24-26 MB .log / .lst files; even the smaller
        # .dck file (~80 KB) adds up when replicated across every checkpoint blob.
        # Downstream consumers only need the *paths* — actual content can be re-read
        # from disk on demand or accessed via the top-level state["dck_content"] field.
        file_version = {
            "version_tag": result.version_tag,
            "dck_path": result.dck_file_path,
            "log_path": result.log_file_path,
            "lst_path": result.lst_file_path,
            "dck_content": "",  # Filtered to reduce checkpoint size
        }

        return {
            "success": result.success,
            "errors": errors,
            "raw_result": result.raw_output,
            "log_content": result.log_content,
            "lst_content": result.lst_content,
            "error_analysis": result.error_analysis.to_dict() if result.error_analysis else None,
            "file_version": file_version,
        }

    def _analyze_errors(
        self,
        errors: List[Dict[str, Any]],
        dck_content: str,
    ) -> Dict[str, Any]:
        """
        Analyze validation errors using sequential LLM calls (with and without search) and fusion.

        This method:
        1. Calls LLM with web search enabled
        2. Calls LLM without web search (local knowledge only)
        3. Fuses the results using a third LLM call

        All calls use self.invoke_agent() to ensure tools are properly passed to LangGraph.

        Note: Sequential execution is used instead of ThreadPoolExecutor to preserve
        LangSmith trace context (contextvars propagation). ThreadPoolExecutor spawns
        new threads that lose the parent trace context, making agent tool calls
        invisible in LangSmith Studio.
        """
        # Build analysis prompt with enhanced JSON format requirements
        prompt = self._build_analysis_prompt(errors, dck_content)

        try:
            # Sequential execution to preserve LangSmith trace context
            # Call 1: LLM with web search
            result_with_search = self._invoke_agent_and_extract(
                prompt, enable_web_search=True
            )

            # Call 2: LLM without web search
            result_without_search = self._invoke_agent_and_extract(
                prompt, enable_web_search=False
            )

            # Call 3: Fuse the two results
            fused_result = self._fuse_analyses(result_with_search, result_without_search)

            logger.info(f"Error analysis completed: {fused_result.get('root_cause', 'Unknown')}")
            return fused_result

        except Exception as e:
            logger.warning(f"Error analysis failed: {e}")
            return {
                "root_cause": "Error analysis failed",
                "responsible_agents": ["validator"],
                "fix_suggestions": ["Please check the DCK format"],
                "priority": "high",
            }

    def _dedupe_and_truncate_errors(
        self,
        errors: List[Dict[str, Any]],
        max_unique: int = 30,
    ) -> List[Dict[str, Any]]:
        """
        Deduplicate validation errors and cap the result at `max_unique` types.

        Used to bound the size of `validation_errors` written to LangGraph
        state. A pathological TRNSYS run (e.g. an illegal-INPUTS configuration
        that fires every timestep) can emit 70k+ near-duplicate errors;
        replicating that list across every checkpoint blob bloats persisted
        state by tens of MB per write. The dedup key and bookkeeping fields
        mirror what `_build_analysis_prompt` already does for the LLM prompt,
        so analysis quality is unchanged — only the *state copy* is shaped.

        Args:
            errors: Raw error list from `_validate_dck` (may be huge).
            max_unique: Maximum number of unique error types to keep.

        Returns:
            A new list of <= `max_unique` + 1 dicts. Each retained error gains
            `repeat_count` and `first_time_h` fields. If truncation occurred,
            an additional sentinel dict (index="_stats", severity="Info") is
            appended carrying the original totals so downstream code can still
            see how many raw errors were produced.
        """
        if not errors:
            return errors  # preserve identity for empty/None input

        unique: Dict[tuple, Dict[str, Any]] = {}
        for e in errors:
            key = (
                e.get('severity', ''),
                e.get('unit_number', 'N/A'),
                e.get('type_number', 'N/A'),
                e.get('detail_message', e.get('message', '')),
            )
            if key not in unique:
                slot = e.copy()
                slot['repeat_count'] = 1
                slot['first_time_h'] = e.get('time_h', '0.000000')
                unique[key] = slot
            else:
                unique[key]['repeat_count'] += 1

        truncated = list(unique.values())[:max_unique]

        if len(errors) > len(truncated):
            truncated.append({
                "index": "_stats",
                "severity": "Info",
                "time_h": "",
                "unit_number": "",
                "type_number": "",
                "detail_message": (
                    f"[stats] total={len(errors)} "
                    f"unique_types={len(unique)} "
                    f"shown={len(truncated)} "
                    f"(truncated to bound state size)"
                ),
                "reported_information": "",
            })

        return truncated

    def _build_analysis_prompt(
        self,
        errors: List[Dict[str, Any]],
        dck_content: str,
    ) -> str:
        """
        Build the analysis prompt with enhanced JSON format requirements.

        Args:
            errors: List of validation errors
            dck_content: DCK file content

        Returns:
            Formatted prompt string
        """
        # Deduplicate and truncate errors to prevent prompt explosion
        MAX_ERRORS_FOR_ANALYSIS = 30  # Maximum unique error types to analyze

        # Deduplicate: group by (severity, unit_number, type_number, detail_message)
        # Only errors with all 4 fields identical are considered duplicates
        unique_errors_dict = {}
        for e in errors:
            # Build unique key from 4 critical fields
            key = (
                e.get('severity', ''),
                e.get('unit_number', 'N/A'),
                e.get('type_number', 'N/A'),
                e.get('detail_message', e.get('message', ''))
            )

            if key not in unique_errors_dict:
                # First occurrence of this error type, record full information
                unique_errors_dict[key] = e.copy()  # Copy original error
                unique_errors_dict[key]['repeat_count'] = 1  # Add repeat counter
                unique_errors_dict[key]['first_time_h'] = e.get('time_h', '0.000000')  # Record first occurrence time
            else:
                # Existing error type, increment counter
                unique_errors_dict[key]['repeat_count'] += 1

        # Convert to list and truncate to first MAX_ERRORS_FOR_ANALYSIS items
        unique_errors_list = list(unique_errors_dict.values())
        errors_to_analyze = unique_errors_list[:MAX_ERRORS_FOR_ANALYSIS]

        # Format error information
        error_lines = []
        for e in errors_to_analyze:
            repeat_count = e.get('repeat_count', 1)
            base_info = (
                f"[{e.get('index', '?')}] {e.get('severity', '')} at time {e.get('time_h', '')}h, "
                f"Unit {e.get('unit_number', 'N/A')} Type {e.get('type_number', 'N/A')}: "
                f"{e.get('detail_message', e.get('message', ''))}"
            )

            # If error repeated, add repeat count annotation
            if repeat_count > 1:
                base_info += f" (repeated {repeat_count} times)"

            error_lines.append(base_info)

        error_text = "\n".join(error_lines)

        # If original error count exceeds truncation limit, add statistics
        if len(errors) > len(errors_to_analyze):
            total_errors = len(errors)
            unique_count = len(unique_errors_list)
            shown_count = len(errors_to_analyze)
            error_text += (
                f"\n\n[INFO] Total {total_errors} errors found, "
                f"{unique_count} unique error types, "
                f"showing first {shown_count} types for analysis."
            )

        prompt = f"""1. 【[CORE★★★★★] Role Definition】
Please analyze the following TRNSYS validation errors:

【【Quick Index — HumanMessage — Error Analysis】】
1. 【[CORE★★★★★] Role Definition】 — Analyze TRNSYS validation errors
2. 【[KEY★★★] Error Information】 — Validation error details from TrnEXE64
3. 【[KEY★★★] DCK Content Excerpt】 — First 2000 characters of the DCK file for context
4. 【[KEY★★★] Your Task】 — 3-step: identify root cause → assign responsible agent → provide repair suggestions
5. 【[CORE★★★★★] Tool Usage Guidance】 — Use relevant tools from 23 available tools to gather information before analysis
6. 【[CORE★★★★★] Output Format】 — JSON: root_cause + responsible_agents + fix_suggestions + priority
7. 【[KEY★★★] Field Requirements】 — Field types, constraints, and valid values for each JSON field
8. 【[KEY★★★] Stop Condition】 — No limit on tool calls; no strict time requirement for JSON output; output ONLY JSON when prompt requirements are fully met

------------------------------------------------------------
2. 【[KEY★★★] Error Information】
{error_text}

------------------------------------------------------------
3. 【[KEY★★★] DCK Content Excerpt】
{dck_content[:2000]}

------------------------------------------------------------
4. 【[KEY★★★] Your Task】
Analyze the errors and provide repair guidance:
1. The root cause of the error
2. Which agent should be responsible for the repair
3. Specific repair suggestions

------------------------------------------------------------
5. 【[CORE★★★★★] Tool Usage Guidance - USE RELEVANT TOOLS FIRST】
Before outputting your analysis, you should use the available tools to gather information based on the error type.
You have 15 tools available across 4 categories. Choose and use the tools that are relevant to the reported errors.

**Available Tools by Category:**

**RAG Tools (7 tools) - Search training data for examples:**
- rag_search_component_specs: Search component specifications from training data
- rag_search_dck_format_examples: Search DCK format examples
- rag_search_connection_examples: Search component connection examples
- rag_search_equation_examples: Search EQUATIONS block examples
- rag_search_parameter_examples: Search parameter configuration examples
- rag_search_input_examples: Search input configuration examples
- rag_search_global_for_dck_and_tmf: Comprehensive search (system + component + format)

**TMF Tools (3 tools) - Verify component specifications:**
- get_tmf_spec: Get complete TMF specification for a component type
- validate_parameter_count: Verify if parameter count matches TMF specification
- validate_input_count: Verify if input count matches TMF specification

**Equation Tools (2 tools) - Verify equation syntax:**
- validate_equation_syntax: Validate equation syntax
- validate_equations_block: Validate entire EQUATIONS block

**DCK Snippet Tools (3 tools) - Search training DCK snippets:**
- search_dck_snippet: Search complete DCK snippet for specific component type
- get_component_special_fields: Analyze special fields (LABELS, External files, etc.)
- rank_components_by_usage: Rank components by usage frequency in training set

**Connection Pattern Tools (3 tools) - Search connection patterns:**
- search_connection_patterns: Search for connection patterns between component types
- get_connection_pattern: Get detailed connection pattern for specific component pair
- find_dcks_with_components: Find training DCK files containing specific components

**Tool Selection Strategy Based on Error Type:**
- **Parameter errors** (count mismatch, value errors) → Use: get_tmf_spec, validate_parameter_count, rag_search_parameter_examples, search_dck_snippet
- **Input/Connection errors** (count mismatch, invalid references) → Use: get_tmf_spec, validate_input_count, rag_search_connection_examples, rag_search_input_examples, search_connection_patterns, get_connection_pattern
- **Component errors** (wrong type, missing component) → Use: get_tmf_spec, rag_search_component_specs, search_dck_snippet, find_dcks_with_components
- **Equation errors** (syntax errors, calculation errors) → Use: validate_equation_syntax, validate_equations_block, rag_search_equation_examples
- **Format errors** (missing fields, incorrect structure) → Use: rag_search_dck_format_examples, get_component_special_fields
- **Unknown/Complex errors** → Use: rag_search_global_for_dck_and_tmf, search_dck_snippet

**Important Guidelines:**
1. You do NOT need to use all 15 tools - select only the tools relevant to the reported errors
2. Use multiple tools to cross-verify information and ensure accuracy
3. Prioritize tools that can provide concrete examples from training data (RAG and DCK Snippet tools)
4. Use TMF tools to verify specifications when parameter/input counts are involved
5. After gathering information from tools, synthesize the findings into your JSON analysis

------------------------------------------------------------
6. 【[CORE★★★★★] Output Format - JSON AFTER USING RELEVANT TOOLS】
After using relevant tools to gather information, output your analysis in valid JSON format.
Do NOT output explanations before or after the JSON.

Output ONLY this JSON structure:
{{
    "root_cause": "Root cause of the error",
    "responsible_agents": ["agent_name"],
    "fix_suggestions": ["Repair suggestion 1", "Repair suggestion 2"],
    "priority": "high/medium/low"
}}

------------------------------------------------------------
7. 【[KEY★★★] Field Requirements】
- "root_cause": String, concise description of the root cause (required)
- "responsible_agents": Array of strings, agent names that should handle repair (required, at least 1)
- "fix_suggestions": Array of strings, specific actionable repair suggestions (required, at least 1)
- "priority": String, one of "high", "medium", or "low" (required)

------------------------------------------------------------
8. 【[KEY★★★] Stop Condition】
- No limit on the maximum number of tool calls; you may call tools as many times as needed to fulfill the prompt requirements
- No strict time requirement for JSON output; output the JSON result once you have completed all prompt requirements (self-judge when ready)
- Output ONLY the JSON result
- Do NOT output explanations before or after JSON
- Do NOT output markdown code blocks
- Do NOT output any other text
- If you cannot parse the error, still output valid JSON with your best analysis
"""

        return prompt

    def _invoke_agent_and_extract(
        self,
        prompt: str,
        enable_web_search: bool = False,
    ) -> Dict[str, Any]:
        """
        Invoke the agent with the prompt and extract JSON result.

        When enable_web_search=True, uses Volcano Engine Responses API with
        web_search tool for online search capability.
        When enable_web_search=False, uses LangGraph agent with local tools.

        Args:
            prompt: The analysis prompt
            enable_web_search: Whether to enable web search via Volcano Engine
                Responses API

        Returns:
            Parsed JSON result

        Raises:
            Exception: If agent invocation or JSON parsing fails
        """
        try:
            if enable_web_search:
                # Use Volcano Engine Responses API with web_search tool
                # Pass validator-specific LLM parameters to match LangGraph agent config:
                #   temperature=0.2 (from EXPERT_TEMPERATURES["validator"])
                #   max_output_tokens=32768 (from create_agent_llm max_tokens_map["validator"])
                #   instructions=self.system_prompt (equivalent to LangGraph agent's prompt)
                messages = [{"role": "user", "content": prompt}]
                raw_response = call_validator_with_search(
                    messages,
                    temperature=0.2,
                    max_output_tokens=32768,
                    instructions=self.system_prompt,
                )
                result_text = self._extract_responses_api_text(raw_response)
                logger.info("Web search analysis completed via Responses API")
            else:
                # Use LangGraph agent with local tools (existing logic)
                task_message = HumanMessage(content=prompt)
                response = self.invoke_agent([task_message])
                result_text = self.extract_result(response)

            # Parse JSON
            parsed_result = self._parse_json_response(result_text)

            logger.info(f"Agent analysis completed (web_search={enable_web_search})")
            return parsed_result

        except Exception as e:
            logger.warning(f"Agent invocation failed (web_search={enable_web_search}): {e}")
            raise

    def _extract_responses_api_text(self, raw_response: Dict[str, Any]) -> str:
        """
        Extract text content from Volcano Engine Responses API response.

        The Responses API returns an 'output' list with items of different types:
        - message (before search): initial LLM response
        - web_search_call: search action
        - message (after search): enhanced response with search results

        We extract the LAST message item's text, which contains the final
        response after web search (if search was triggered).
        See: 火山方舟-联网搜索-Web-Search.md - "模型输出" section.

        Args:
            raw_response: Raw JSON response from Volcano Engine Responses API

        Returns:
            Extracted text string

        Raises:
            ValueError: If text cannot be extracted from the response
        """
        output = raw_response.get("output", [])

        # Find the LAST message item (contains the final response after web search)
        last_message_text = None
        for item in output:
            if item.get("type") == "message":
                content = item.get("content", [])
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "output_text":
                        text = block.get("text", "")
                        if text:
                            last_message_text = text

        if last_message_text:
            return last_message_text

        # Fallback: try to find any text content in output items
        for item in output:
            content = item.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and "text" in block:
                        text = block["text"]
                        if text:
                            return text
            elif isinstance(content, str) and content:
                return content

        raise ValueError(
            f"Could not extract text from Responses API response: "
            f"{json.dumps(raw_response, ensure_ascii=False)[:500]}"
        )

    def _parse_json_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse JSON from response text.

        Handles multiple JSON formats:
        - Plain JSON
        - JSON in markdown code blocks
        - JSON with surrounding text

        Args:
            response_text: Response text from LLM

        Returns:
            Parsed JSON dictionary

        Raises:
            ValueError: If JSON cannot be parsed
        """
        # Try to extract JSON from markdown code blocks first
        json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find JSON object directly
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                json_str = response_text

        # Parse JSON
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON: {e}")
            logger.debug(f"Response text: {response_text[:500]}")
            raise ValueError(f"Invalid JSON response: {e}")

        # Validate and normalize fields
        if not isinstance(data, dict):
            raise ValueError(f"Expected JSON object, got {type(data)}")

        result = {
            "root_cause": str(data.get("root_cause", "Unknown error")),
            "responsible_agents": data.get("responsible_agents", ["validator"]),
            "fix_suggestions": data.get("fix_suggestions", ["Please check the DCK format"]),
            "priority": data.get("priority", "high"),
        }

        # Validate field types
        if not isinstance(result["responsible_agents"], list):
            result["responsible_agents"] = ["validator"]
        if not isinstance(result["fix_suggestions"], list):
            result["fix_suggestions"] = ["Please check the DCK format"]
        if result["priority"] not in ["high", "medium", "low"]:
            result["priority"] = "high"

        # Ensure non-empty lists
        if not result["responsible_agents"]:
            result["responsible_agents"] = ["validator"]
        if not result["fix_suggestions"]:
            result["fix_suggestions"] = ["Please check the DCK format"]

        return result

    def _fuse_analyses(
        self,
        result_with_search: Dict[str, Any],
        result_without_search: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Fuse two analysis results using LLM.

        Args:
            result_with_search: Analysis result with web search
            result_without_search: Analysis result without web search

        Returns:
            Fused analysis result
        """
        try:
            # Build fusion prompt
            fusion_prompt = f"""1. 【[CORE★★★★★] Role Definition】
You have two error analyses from different sources, please fuse two error analyses into one comprehensive result:

【【Quick Index — HumanMessage — Fusion】】
1. 【[CORE★★★★★] Role Definition】 — Fuse two error analyses into one comprehensive result
2. 【[KEY★★★] Analysis 1 - With Web Search】 — Error analysis result from LLM with web search enabled
3. 【[KEY★★★] Analysis 2 - Without Web Search】 — Error analysis result from LLM with local knowledge only
4. 【[KEY★★★] Fusion Task】 — 4-step: combine root causes → merge agents → merge suggestions → choose priority
5. 【[CORE★★★★★] Output Format - ONLY JSON】 — JSON: root_cause + responsible_agents + fix_suggestions + priority
6. 【[KEY★★★] Stop Condition】 — No limit on tool calls; no strict time requirement for JSON output; output ONLY JSON when prompt requirements are fully met

------------------------------------------------------------
2. 【[KEY★★★] Analysis 1 - With Web Search】
{json.dumps(result_with_search, indent=2, ensure_ascii=False)}

------------------------------------------------------------
3. 【[KEY★★★] Analysis 2 - Without Web Search】
{json.dumps(result_without_search, indent=2, ensure_ascii=False)}

------------------------------------------------------------
4. 【[KEY★★★] Fusion Task】
Please fuse these analyses into one comprehensive analysis:
1. Combine root causes if they're complementary, or choose the most accurate one
2. Merge responsible agents (remove duplicates)
3. Merge fix suggestions (remove duplicates, keep the most actionable ones)
4. Choose the highest priority level

------------------------------------------------------------
5. 【[CORE★★★★★] Output Format - ONLY JSON】
Output ONLY valid JSON in this format:
{{
    "root_cause": "...",
    "responsible_agents": [...],
    "fix_suggestions": [...],
    "priority": "high/medium/low"
}}

------------------------------------------------------------
6. 【[KEY★★★] Stop Condition】
- No limit on the maximum number of tool calls; you may call tools as many times as needed to fulfill the prompt requirements
- No strict time requirement for JSON output; output the JSON result once you have completed all prompt requirements (self-judge when ready)
- Do NOT output any explanations or other text.
"""

            # Invoke agent for fusion
            task_message = HumanMessage(content=fusion_prompt)
            response = self.invoke_agent([task_message])
            result_text = self.extract_result(response)

            # Parse fused result
            fused_result = self._parse_json_response(result_text)

            logger.info("Analysis fusion completed")
            return fused_result

        except Exception as e:
            logger.warning(f"Analysis fusion failed: {e}")
            # Fallback: simple merge
            return self._simple_merge_analyses(result_with_search, result_without_search)

    def _simple_merge_analyses(
        self,
        result1: Dict[str, Any],
        result2: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Simple merge of two analysis results (used as fallback).

        Args:
            result1: First analysis result
            result2: Second analysis result

        Returns:
            Merged result
        """
        # Merge responsible agents (remove duplicates)
        agents = list(set(
            result1.get("responsible_agents", []) +
            result2.get("responsible_agents", [])
        ))
        if not agents:
            agents = ["validator"]

        # Merge fix suggestions (remove duplicates)
        suggestions = list(set(
            result1.get("fix_suggestions", []) +
            result2.get("fix_suggestions", [])
        ))
        if not suggestions:
            suggestions = ["Please check the DCK format"]

        # Choose higher priority
        priority_order = {"low": 0, "medium": 1, "high": 2}
        p1 = priority_order.get(result1.get("priority", "low"), 0)
        p2 = priority_order.get(result2.get("priority", "low"), 0)
        priority = max(p1, p2)
        priority_str = {0: "low", 1: "medium", 2: "high"}.get(priority, "high")

        # Prefer result1's root cause, fallback to result2
        root_cause = result1.get("root_cause") or result2.get("root_cause") or "Unknown error"

        return {
            "root_cause": root_cause,
            "responsible_agents": agents,
            "fix_suggestions": suggestions,
            "priority": priority_str,
        }

    def _record_success(self, state: Dict[str, Any], dck_content: str):
        """Record successful generation to Memory (disabled - memory module removed)."""
        try:
            # Memory module removed - success recording disabled
            logger.debug("Success recording disabled - memory module has been removed")
        except Exception as e:
            logger.warning(f"Failed to record success: {e}")

    def _record_error_fix(
        self,
        state: Dict[str, Any],
        error_message: str,
        fix_description: str,
    ):
        """
        Record error fix experience to Memory for future learning.

        Based on Mem0 official documentation - stores error patterns and fixes
        to help avoid similar errors in future generations.

        Args:
            state: Current state containing component information.
            error_message: The error that was encountered.
            fix_description: How the error was fixed or should be fixed.
        """
        try:
            # Memory module removed - error fix recording disabled
            logger.debug("Error fix recording disabled - memory module has been removed")
        except Exception as e:
            logger.warning(f"Failed to record error fix: {e}")

    def validate_and_repair(
        self,
        state: Dict[str, Any],
        max_iterations: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Validate with iterative repair.

        Args:
            state: Current state.
            max_iterations: Maximum repair iterations.

        Returns:
            Final state after validation/repair.
        """
        if max_iterations is None:
            from ...core.performance_config import get_performance_config
            max_iterations = get_performance_config().max_workflow_iterations

        for iteration in range(max_iterations):
            result = self.process(state)

            if result.get("validation_result") == "success":
                return result

            # Update state for next iteration
            state.update(result)
            state["iteration"] = iteration + 1

            logger.info(f"Validation iteration {iteration + 1} failed, continuing...")

        return result

    def _fix_missing_special_fields(self, components: List[Dict[str, Any]], task_description: str = "") -> List[Dict[str, Any]]:
        """
        Compare components with training DCK snippets and fix missing special fields.

        This method implements the conservative approach:
        1. Keep the current architecture (generate from TMF spec)
        2. Compare with training DCK snippets
        3. Fix missing fields (LABELS, External files, etc.) based on training data statistics
        4. Extract file paths from task_description if available

        Args:
            components: List of component dictionaries.
            task_description: Task description to extract file paths from.

        Returns:
            Updated components with missing special fields added.
        """
        if not DCK_SNIPPET_TOOLS_AVAILABLE:
            logger.debug("DCK snippet tools not available, skipping special fields fix")
            return components

        # Extract file paths from task description
        extracted_paths = self._extract_file_paths_from_task(task_description)

        updated_components = []

        for comp in components:
            type_number = comp.get("type_number", "")
            if not type_number:
                updated_components.append(comp)
                continue

            try:
                # Get special fields statistics from training data
                special_fields = get_special_fields_for_type(type_number)

                # Apply missing special fields with extracted paths
                comp = self._apply_special_fields(comp, special_fields, extracted_paths)

            except Exception as e:
                logger.debug(f"Could not get special fields for {type_number}: {e}")

            updated_components.append(comp)

        return updated_components

    def _extract_file_paths_from_task(self, task_description: str) -> Dict[str, str]:
        """
        Extract file paths from task description.

        Args:
            task_description: Task description text.

        Returns:
            Dictionary with extracted paths: {"weather_file": "...", "building_file": "..."}
        """
        extracted = {}

        if not task_description:
            return extracted

        # Extract weather file path (Type15)
        # Look for patterns like: "d:\TRNSYS18_03\Weather\...\xxx.tm2"
        weather_patterns = [
            r'["\']?(d:\\[^"\']+\.tm2)["\']?',
            r'["\']?(d:/[^"\']+\.tm2)["\']?',
            r'located at["\']?([^"\']+\.tm2)["\']?',
            r'weather[^"\']*["\']([^"\']+\.tm2)["\']',
        ]
        for pattern in weather_patterns:
            match = re.search(pattern, task_description, re.IGNORECASE)
            if match:
                extracted["weather_file"] = match.group(1).replace("/", "\\")
                logger.info(f"Extracted weather file from task: {extracted['weather_file']}")
                break

        # Extract building file path (Type56)
        # Look for patterns like: "Building_step2.b18" or "xxx.b18"
        building_patterns = [
            r'["\']?([A-Za-z0-9_-]+\.b18)["\']?',
            r'located at["\']?([^"\']+\.b18)["\']?',
            r'building[^"\']*["\']([^"\']+\.b18)["\']',
        ]
        for pattern in building_patterns:
            match = re.search(pattern, task_description, re.IGNORECASE)
            if match:
                extracted["building_file"] = match.group(1)
                logger.info(f"Extracted building file from task: {extracted['building_file']}")
                break

        return extracted

    def _apply_special_fields(self, comp: Dict[str, Any], special_fields: Dict[str, Any], extracted_paths: Dict[str, str] = None) -> Dict[str, Any]:
        """
        Apply missing special fields to a component based on training data statistics.

        Args:
            comp: Component dictionary.
            special_fields: Special fields statistics from training data.
            extracted_paths: Extracted file paths from task description.

        Returns:
            Updated component dictionary.
        """
        if extracted_paths is None:
            extracted_paths = {}

        # Track if we need to rebuild the dict
        needs_rebuild = False
        new_fields = {}

        # Check if component needs LABELS (based on training data frequency)
        if special_fields.get("has_labels", False):
            if not comp.get("labels"):
                default_labels = special_fields.get("default_labels", [])
                if default_labels:
                    new_fields["labels"] = default_labels
                    needs_rebuild = True
                    logger.info(f"Added LABELS to {comp.get('type_number')}: {default_labels}")

        # Check if component needs External files (based on training data frequency)
        if special_fields.get("has_external_files", False):
            type_num = comp.get("type_number", "")
            if not comp.get("external_files"):
                # For External files, we need more context (file path depends on user's system)
                # So we only add a placeholder if the component type usually has external files
                # The actual path should be provides by the user or other agents
                external_files_pct = special_fields.get("external_files_pct", 0)
                if external_files_pct > 80:
                    # High frequency - this component type almost always has external files
                    # Add a placeholder that will need to be filled
                    default_logical_unit = special_fields.get("default_logical_unit", 30)

                    # Find logical unit from parameters by searching for parameter name containing "Logical unit"
                    params = comp.get("parameters", [])
                    logical_unit_found = False
                    for param in params:
                        if isinstance(param, dict):
                            param_name = param.get("name", "").lower()
                            if "logical unit" in param_name or "logical_unit" in param_name:
                                try:
                                    default_logical_unit = int(param.get("value", default_logical_unit))
                                    logical_unit_found = True
                                    logger.debug(f"Found Logical unit from parameter '{param.get('name')}': {default_logical_unit}")
                                    break
                                except (ValueError, TypeError):
                                    pass

                    # Determine file path based on component type, using extracted paths if available
                    file_path = self._get_file_path_for_type(type_num, comp, extracted_paths)

                    if file_path:
                        new_fields["external_files"] = [{
                            "path": file_path,
                            "logical_unit": default_logical_unit,
                            "prompt": self._get_file_prompt(type_num),
                        }]
                        needs_rebuild = True
                        logger.info(f"Added External files to {type_num}: {file_path} (LU={default_logical_unit})")
            else:
                # Component already has external_files, but check if prompt is missing
                # This can happen when parameter_configurator provides external_files without prompt
                external_files = comp.get("external_files", [])
                updated_external_files = []
                modified = False
                for ef in external_files:
                    if isinstance(ef, dict) and not ef.get("prompt"):
                        ef_copy = dict(ef)
                        ef_copy["prompt"] = self._get_file_prompt(type_num)
                        updated_external_files.append(ef_copy)
                        modified = True
                        logger.info(f"Added missing prompt to external_files for {type_num}")
                    else:
                        updated_external_files.append(ef)

                if modified:
                    new_fields["external_files"] = updated_external_files
                    needs_rebuild = True

        # CRITICAL: Rebuild component dict with unit_number first if we added new fields
        if needs_rebuild:
            unit_num = comp.get("unit_number")
            new_comp = {"unit_number": unit_num}
            for key, value in comp.items():
                if key != "unit_number":
                    new_comp[key] = value
            # Apply new fields
            for key, value in new_fields.items():
                new_comp[key] = value
            return new_comp

        return comp

    def _get_file_path_for_type(self, type_number: str, comp: Dict[str, Any], extracted_paths: Dict[str, str]) -> str:
        """
        Get file path for a component type, preferring extracted paths from task description.

        Args:
            type_number: Component type number.
            comp: Component dictionary.
            extracted_paths: Extracted file paths from task description.

        Returns:
            File path string, or empty string if not applicable.
        """
        type_lower = str(type_number).lower()

        # Type15 - Weather data file
        if "15" in type_lower:
            # Prefer extracted path from task description
            if extracted_paths.get("weather_file"):
                return extracted_paths["weather_file"]
            # Fallback to default
            return "d:\\TRNSYS18_03\\Weather\\US-TMY2\\US-WI-Madison-14837.tm2"

        # Type56 - Building description file
        if "56" in type_lower:
            # Prefer extracted path from task description
            if extracted_paths.get("building_file"):
                return extracted_paths["building_file"]
            # Fallback to default
            return "Building.b18"

        # Type9 - Data reader file
        if type_lower.startswith("9") or type_lower.startswith("type9"):
            return "data.txt"

        return ""

    def _get_file_prompt(self, type_number: str) -> str:
        """
        Get file prompt for a component type.

        Args:
            type_number: Component type number.

        Returns:
            File prompt string.
        """
        return "External data file"

    # =========================================================================
    # DERIVATIVES block support for thermal storage components
    # (Type4a~4f, Type156, Type158, Type60a~60x + Type60cNoHeat)
    # =========================================================================

    # TMF filename suffixes that uniquely identify thermal storage components
    # IMPORTANT: use full TMF filenames for exact matching to avoid
    # false positives (e.g. Type160a.tmf should NOT match Type60a.tmf)
    STORAGE_TMF_IDS = {
        "Type4a.tmf", "Type4b.tmf", "Type4c.tmf",
        "Type4d.tmf", "Type4e.tmf", "Type4f.tmf",
        "Type156.tmf", "Type158.tmf",
    }

    # Type60 series thermal storage components (25 variants)
    STORAGE_TYPE60_TMF_IDS = {
        "Type60a.tmf", "Type60b.tmf", "Type60c.tmf", "Type60cNoHeat.tmf",
        "Type60d.tmf", "Type60e.tmf", "Type60f.tmf", "Type60g.tmf",
        "Type60h.tmf", "Type60i.tmf", "Type60j.tmf", "Type60k.tmf",
        "Type60l.tmf", "Type60m.tmf", "Type60n.tmf", "Type60o.tmf",
        "Type60p.tmf", "Type60q.tmf", "Type60r.tmf", "Type60s.tmf",
        "Type60t.tmf", "Type60u.tmf", "Type60v.tmf", "Type60w.tmf",
        "Type60x.tmf",
    }

    # Non-Uniform Type60 variants: node count can be inferred from parameters
    # (parameter count = 44 + 2 * node_count; each node adds "Height of node"
    #  and "Additional loss coefficient for node" parameters)
    STORAGE_TYPE60_NON_UNIFORM_TMF_IDS = {
        "Type60e.tmf", "Type60f.tmf", "Type60g.tmf", "Type60h.tmf",
        "Type60m.tmf", "Type60n.tmf", "Type60o.tmf", "Type60p.tmf",
        "Type60u.tmf", "Type60v.tmf", "Type60w.tmf", "Type60x.tmf",
    }

    def _is_storage_component(self, comp: Dict[str, Any]) -> bool:
        """
        Check if a component is one of the thermal storage types
        (Type4a~4f, Type156, Type158, Type60a~60x, Type60cNoHeat)
        by examining the TMF filename in model_path.

        Uses exact TMF filename matching to avoid false positives
        (e.g. Type160a.tmf will NOT match Type60a.tmf).

        Args:
            comp: Component dictionary.

        Returns:
            True if the component is a thermal storage type.
        """
        model_path = comp.get("model_path", "")
        if not model_path:
            return False
        # Extract the TMF filename (last segment after backslash or forward slash)
        tmf_filename = model_path.replace("/", "\\").rsplit("\\", 1)[-1].strip()
        return (tmf_filename in self.STORAGE_TMF_IDS
                or tmf_filename in self.STORAGE_TYPE60_TMF_IDS)

    def _get_storage_tmf_id(self, comp: Dict[str, Any]) -> str:
        """
        Get the TMF ID (e.g. 'Type4a', 'Type60a', 'Type60cNoHeat') from model_path.

        Args:
            comp: Component dictionary.

        Returns:
            TMF ID string (e.g. 'Type4a', 'Type156', 'Type60a'), or empty string.
        """
        model_path = comp.get("model_path", "")
        if not model_path:
            return ""
        tmf_filename = model_path.replace("/", "\\").rsplit("\\", 1)[-1].strip()
        if tmf_filename in self.STORAGE_TMF_IDS or tmf_filename in self.STORAGE_TYPE60_TMF_IDS:
            return tmf_filename.replace(".tmf", "")
        return ""

    def _determine_node_count(self, comp: Dict[str, Any], tmf_id: str) -> int:
        """
        Determine the number of nodes for a thermal storage component
        by inspecting its parameters.

        For Type4a~4f: count parameters whose name contains 'Height of node'.
        For Type156/Type158: read the value of the parameter named
            'Number of tank nodes'.
        For Type60 Non-Uniform variants (e,f,g,h,m,n,o,p,u,v,w,x):
            count parameters whose name contains 'Height of node'.
        For Type60 Uniform variants (a,b,c,cNoHeat,d,i,j,k,l,q,r,s,t):
            cannot determine from parameters alone, returns -1 to signal
            that LLM or default should be used.

        Args:
            comp: Component dictionary.
            tmf_id: TMF ID string (e.g. 'Type4a', 'Type156', 'Type60a').

        Returns:
            Number of nodes (positive int), or -1 if unable to determine
            from parameters (Type60 Uniform variants).
        """
        params = comp.get("parameters", [])

        if tmf_id.lower() in ("type156", "type158"):
            # Parameter 3 is "Number of tank nodes"
            for p in params:
                if isinstance(p, dict):
                    name = p.get("name", "").lower()
                    if "number of tank nodes" in name:
                        try:
                            return int(float(p.get("value", 1)))
                        except (ValueError, TypeError):
                            pass
            # Fallback: if parameter list is long enough, try index 2 (0-based)
            if len(params) >= 3:
                p = params[2]
                if isinstance(p, dict):
                    try:
                        return int(float(p.get("value", 1)))
                    except (ValueError, TypeError):
                        pass
            return 1

        # Check if this is a Type60 component
        tmf_filename = tmf_id + ".tmf"
        if tmf_filename in self.STORAGE_TYPE60_TMF_IDS:
            # Type60 Non-Uniform variants: count "Height of node" parameters
            if tmf_filename in self.STORAGE_TYPE60_NON_UNIFORM_TMF_IDS:
                count = 0
                for p in params:
                    if isinstance(p, dict):
                        name = p.get("name", "").lower()
                        if "height of node" in name:
                            count += 1
                if count > 0:
                    return count
                # Fallback: try (total_params - 44) / 2
                if len(params) > 44:
                    inferred = (len(params) - 44) // 2
                    if inferred > 0:
                        return inferred
                # Cannot determine — signal for LLM/default
                return -1
            else:
                # Type60 Uniform variants: cannot determine from parameters
                return -1

        # Type4a~4f: count "Height of node" parameters
        count = 0
        for p in params:
            if isinstance(p, dict):
                name = p.get("name", "").lower()
                if "height of node" in name:
                    count += 1
        return count if count > 0 else 1

    def _get_initial_temperatures_via_llm(
        self,
        storage_components: List[Dict[str, Any]],
        node_counts: Dict[int, int],
        task_description: str,
    ) -> Dict[int, List[float]]:
        """
        Use LLM to analyze the task description and extract initial node
        temperatures for thermal storage components.

        If the task description mentions initial temperatures, the LLM
        extracts them. Otherwise all nodes default to 20 degC.

        Retry mechanism: up to 3 attempts if LLM returns invalid format.
        After 3 failures, fall back to 20 degC for all nodes.

        Args:
            storage_components: List of storage component dicts.
            node_counts: Mapping of unit_number -> node count.
            task_description: Original task description.

        Returns:
            Mapping of unit_number -> list of temperatures (one per node).
        """
        default_temps = {
            comp.get("unit_number", 0): [20.0] * node_counts.get(
                comp.get("unit_number", 0), 1
            )
            for comp in storage_components
        }

        # Build a summary of which units need temperatures
        unit_info_lines = []
        for comp in storage_components:
            unit_num = comp.get("unit_number", 0)
            unit_name = comp.get("unit_name", "")
            tmf_id = self._get_storage_tmf_id(comp)
            n_nodes = node_counts.get(unit_num, 1)
            unit_info_lines.append(
                f"UNIT {unit_num} ({unit_name}, {tmf_id}): {n_nodes} nodes"
            )
        unit_info_text = "\n".join(unit_info_lines)

        # Build a concrete example showing the exact expected output
        example_entries = []
        for comp in storage_components:
            unit_num = comp.get("unit_number", 0)
            n_nodes = node_counts.get(unit_num, 1)
            example_vals = ", ".join(["20.0"] * n_nodes)
            example_entries.append(f'    "{unit_num}": [{example_vals}]')
        example_json = (
            '{\n  "unit_temperatures": {\n'
            + ",\n".join(example_entries)
            + "\n  }\n}"
        )

        prompt = f"""Analyze the following TRNSYS task description and extract initial
temperatures for thermal storage tank nodes.

[Task Description]
{task_description if task_description else '(No task description provided)'}

[Storage Components Requiring DERIVATIVES]
{unit_info_text}

[STRICT FORMAT RULES - YOU MUST FOLLOW EXACTLY]
1. Your response MUST contain ONLY a single valid JSON object. No explanation,
   no commentary, no markdown code fences, no text before or after the JSON.
2. The JSON MUST have exactly one top-level key: "unit_temperatures".
3. Under "unit_temperatures", there MUST be one entry per storage component
   listed above. The key MUST be the unit number as a STRING (e.g. "5").
4. Each value MUST be a JSON array of numbers. The array length MUST exactly
   equal the node count shown above for that unit. Every element MUST be a
   valid number (integer or float).
5. If the task description specifies initial node temperatures, use those
   values. If it does NOT mention initial temperatures, use 20 for every node.
6. Do NOT add any extra keys, nested objects, or comments inside the JSON.

[EXAMPLE - This is the EXACT format you must produce]
{example_json}
"""

        system_msg = SystemMessage(
            content=(
                "You are a TRNSYS thermal storage expert. "
                "Your ONLY task is to output a JSON object with initial node "
                "temperatures. You MUST strictly follow the format rules in the "
                "user message. Output ONLY the raw JSON object — no explanation, "
                "no markdown, no code fences, no extra text whatsoever."
            )
        )

        # Retry up to 3 times if LLM returns invalid format
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                ai_message = self.llm.invoke(
                    [system_msg, HumanMessage(content=prompt)]
                )
                result_text = (
                    ai_message.content
                    if isinstance(ai_message.content, str)
                    else str(ai_message.content)
                )

                # Parse JSON from response (handle markdown code blocks too)
                json_str = None
                code_block_match = re.search(
                    r'```(?:json)?\s*(.*?)\s*```', result_text, re.DOTALL
                )
                if code_block_match:
                    json_str = code_block_match.group(1)
                else:
                    json_obj_match = re.search(
                        r'\{.*\}', result_text, re.DOTALL
                    )
                    if json_obj_match:
                        json_str = json_obj_match.group(0)

                if not json_str:
                    logger.warning(
                        f"LLM temperature extraction attempt {attempt}/{max_attempts}: "
                        f"no JSON found in response"
                    )
                    continue

                data = json.loads(json_str)
                unit_temps = data.get("unit_temperatures", {})

                if not unit_temps:
                    logger.warning(
                        f"LLM temperature extraction attempt {attempt}/{max_attempts}: "
                        f"'unit_temperatures' key missing or empty"
                    )
                    continue

                # Validate that every storage component has correct node count
                result = {}
                all_valid = True
                for comp in storage_components:
                    unit_num = comp.get("unit_number", 0)
                    n_nodes = node_counts.get(unit_num, 1)
                    temps = unit_temps.get(str(unit_num), None)

                    if (
                        temps is not None
                        and isinstance(temps, list)
                        and len(temps) == n_nodes
                        and all(isinstance(t, (int, float)) for t in temps)
                    ):
                        result[unit_num] = [float(t) for t in temps]
                    else:
                        all_valid = False
                        logger.warning(
                            f"LLM temperature extraction attempt {attempt}/{max_attempts}: "
                            f"UNIT {unit_num} expected {n_nodes} numeric values, "
                            f"got {temps!r}"
                        )
                        break

                if all_valid and result:
                    logger.info(
                        f"LLM temperature extraction succeeded on attempt {attempt}"
                    )
                    return result

            except (json.JSONDecodeError, ValueError, TypeError) as e:
                logger.warning(
                    f"LLM temperature extraction attempt {attempt}/{max_attempts} "
                    f"parse error: {e}"
                )
            except Exception as e:
                logger.warning(
                    f"LLM temperature extraction attempt {attempt}/{max_attempts} "
                    f"unexpected error: {e}"
                )

        # All attempts exhausted — fall back to 20 degC for all nodes
        logger.warning(
            f"LLM temperature extraction failed after {max_attempts} attempts, "
            f"using default 20 degC for all nodes"
        )
        return default_temps

    def _build_derivatives_block(
        self,
        n_nodes: int,
        temperatures: List[float],
        tmf_id: str,
    ) -> str:
        """
        Build a DERIVATIVES text block for a thermal storage component.

        Args:
            n_nodes: Number of nodes.
            temperatures: List of initial temperatures (one per node).
            tmf_id: TMF ID (e.g. 'Type4a', 'Type156') to choose comment style.

        Returns:
            DERIVATIVES block string (without trailing newline).
        """
        # Choose comment label based on component type
        if tmf_id.lower() in ("type156", "type158"):
            label_prefix = "Initial tank temperature for node"
        else:
            label_prefix = "Initial temperature of node"

        lines = [f"DERIVATIVES {n_nodes}"]
        for i in range(1, n_nodes + 1):
            temp = temperatures[i - 1] if i - 1 < len(temperatures) else 20.0
            # Format: remove trailing zeros for clean output
            temp_str = f"{temp:g}"
            lines.append(f"{temp_str}\t\t! {i} {label_prefix}-{i}")

        return "\n".join(lines)

    def _add_derivatives_for_storage_tanks(
        self,
        dck_content: str,
        components: List[Dict[str, Any]],
        task_description: str,
    ) -> str:
        """
        Post-process assembled DCK content to add DERIVATIVES blocks
        for thermal storage components (Type4a~4f, Type156, Type158,
        Type60a~60x, Type60cNoHeat).

        This method:
        1. Identifies storage components by TMF filename in model_path
        2. Determines node count from parameters (or via LLM for Type60
           Uniform variants that cannot be inferred from parameters)
        3. Uses LLM to extract initial temperatures from task description
        4. Inserts DERIVATIVES blocks after INITIAL INPUT VALUES lines

        Args:
            dck_content: Assembled DCK content string.
            components: List of component dictionaries from state.
            task_description: Original task description.

        Returns:
            DCK content with DERIVATIVES blocks added.
        """
        # Step 1: Identify storage components
        storage_components = [c for c in components if self._is_storage_component(c)]
        if not storage_components:
            logger.debug("No thermal storage components found, skipping DERIVATIVES")
            return dck_content

        logger.info(
            f"Found {len(storage_components)} thermal storage component(s), "
            f"adding DERIVATIVES blocks"
        )

        # Step 2: Determine node counts
        node_counts = {}
        # Collect Type60 components that need LLM to determine node count
        type60_need_llm_node_count = []
        for comp in storage_components:
            unit_num = comp.get("unit_number", 0)
            tmf_id = self._get_storage_tmf_id(comp)
            n_nodes = self._determine_node_count(comp, tmf_id)
            if n_nodes == -1:
                # Type60 Uniform variant: cannot determine from parameters
                type60_need_llm_node_count.append(comp)
                logger.info(
                    f"  UNIT {unit_num} ({tmf_id}): node count unknown from "
                    f"parameters, will use LLM or default"
                )
            else:
                node_counts[unit_num] = n_nodes
                logger.info(
                    f"  UNIT {unit_num} ({tmf_id}): {n_nodes} nodes "
                    f"(from parameters)"
                )

        # Step 2b: For Type60 Uniform variants, use LLM to determine node count
        if type60_need_llm_node_count:
            llm_node_counts = self._determine_type60_node_counts_via_llm(
                type60_need_llm_node_count, task_description
            )
            for comp in type60_need_llm_node_count:
                unit_num = comp.get("unit_number", 0)
                tmf_id = self._get_storage_tmf_id(comp)
                n_nodes = llm_node_counts.get(unit_num, 5)
                node_counts[unit_num] = n_nodes
                logger.info(
                    f"  UNIT {unit_num} ({tmf_id}): {n_nodes} nodes "
                    f"(from LLM/default)"
                )

        # Step 3: Get initial temperatures via LLM
        temperatures = self._get_initial_temperatures_via_llm(
            storage_components, node_counts, task_description
        )

        # Step 4: Insert DERIVATIVES blocks into DCK content
        # Strategy: for each storage component, find its UNIT block in the DCK
        # text and insert DERIVATIVES after the INITIAL INPUT VALUES line.
        lines = dck_content.split("\n")
        result_lines = []
        i = 0

        # Build a lookup: unit_number -> (tmf_id, n_nodes, temps)
        unit_lookup = {}
        for comp in storage_components:
            unit_num = comp.get("unit_number", 0)
            tmf_id = self._get_storage_tmf_id(comp)
            n_nodes = node_counts.get(unit_num, 1)
            temps = temperatures.get(unit_num, [20.0] * n_nodes)
            unit_lookup[unit_num] = (tmf_id, n_nodes, temps)

        current_unit_num = None

        while i < len(lines):
            line = lines[i]
            result_lines.append(line)

            # Track which UNIT block we are inside
            unit_match = re.match(r'^UNIT\s+(\d+)\s+TYPE\s+', line)
            if unit_match:
                current_unit_num = int(unit_match.group(1))

            # Detect "*** INITIAL INPUT VALUES" line for a storage component
            if (
                current_unit_num is not None
                and current_unit_num in unit_lookup
                and line.strip() == "*** INITIAL INPUT VALUES"
            ):
                # The next line is the actual initial values line
                i += 1
                if i < len(lines):
                    result_lines.append(lines[i])  # append the initial values line

                    # Now insert the DERIVATIVES block
                    tmf_id, n_nodes, temps = unit_lookup[current_unit_num]
                    deriv_block = self._build_derivatives_block(n_nodes, temps, tmf_id)
                    result_lines.append(deriv_block)

                    logger.info(
                        f"  Inserted DERIVATIVES block for UNIT {current_unit_num} "
                        f"({tmf_id}, {n_nodes} nodes)"
                    )

                    # Remove from lookup so we don't insert twice
                    del unit_lookup[current_unit_num]

            i += 1

        return "\n".join(result_lines)

    def _determine_type60_node_counts_via_llm(
        self,
        type60_components: List[Dict[str, Any]],
        task_description: str,
    ) -> Dict[int, int]:
        """
        Use LLM to analyze the task description and determine node counts
        for Type60 Uniform variants whose node count cannot be inferred
        from parameters.

        If the task description mentions node counts for these components,
        the LLM extracts them. Otherwise defaults to 5 nodes.

        Retry mechanism: up to 3 attempts. After 3 failures, fall back to 5.

        Args:
            type60_components: List of Type60 component dicts that need
                LLM-based node count determination.
            task_description: Original task description.

        Returns:
            Mapping of unit_number -> node count.
        """
        # Default: 5 nodes for all
        default_counts = {
            comp.get("unit_number", 0): 5
            for comp in type60_components
        }

        if not task_description:
            logger.info(
                "No task description provided, using default 5 nodes "
                "for Type60 Uniform variants"
            )
            return default_counts

        # Build a summary of which units need node counts
        unit_info_lines = []
        for comp in type60_components:
            unit_num = comp.get("unit_number", 0)
            unit_name = comp.get("unit_name", "")
            tmf_id = self._get_storage_tmf_id(comp)
            unit_info_lines.append(
                f"UNIT {unit_num} ({unit_name}, {tmf_id})"
            )
        unit_info_text = "\n".join(unit_info_lines)

        # Build example JSON
        example_entries = []
        for comp in type60_components:
            unit_num = comp.get("unit_number", 0)
            example_entries.append(f'    "{unit_num}": 5')
        example_json = (
            '{\n  "unit_node_counts": {\n'
            + ",\n".join(example_entries)
            + "\n  }\n}"
        )

        prompt = f"""Analyze the following TRNSYS task description and determine the
number of thermal storage tank nodes for each Type60 component listed below.

[Task Description]
{task_description}

[Type60 Components Requiring Node Count]
{unit_info_text}

[STRICT FORMAT RULES - YOU MUST FOLLOW EXACTLY]
1. Your response MUST contain ONLY a single valid JSON object. No explanation,
   no commentary, no markdown code fences, no text before or after the JSON.
2. The JSON MUST have exactly one top-level key: "unit_node_counts".
3. Under "unit_node_counts", there MUST be one entry per component listed above.
   The key MUST be the unit number as a STRING (e.g. "5").
4. Each value MUST be a positive integer representing the number of tank nodes.
5. If the task description specifies the number of nodes for a component, use
   that value. If it does NOT mention the number of nodes, use 5 as default.
6. Do NOT add any extra keys, nested objects, or comments inside the JSON.

[EXAMPLE - This is the EXACT format you must produce]
{example_json}
"""

        system_msg = SystemMessage(
            content=(
                "You are a TRNSYS thermal storage expert. "
                "Your ONLY task is to output a JSON object with node counts "
                "for Type60 thermal storage components. "
                "You MUST strictly follow the format rules in the user message. "
                "Output ONLY the raw JSON object — no explanation, "
                "no markdown, no code fences, no extra text whatsoever."
            )
        )

        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                ai_message = self.llm.invoke(
                    [system_msg, HumanMessage(content=prompt)]
                )
                result_text = (
                    ai_message.content
                    if isinstance(ai_message.content, str)
                    else str(ai_message.content)
                )

                # Parse JSON from response
                json_str = None
                code_block_match = re.search(
                    r'```(?:json)?\s*(.*?)\s*```', result_text, re.DOTALL
                )
                if code_block_match:
                    json_str = code_block_match.group(1)
                else:
                    json_obj_match = re.search(
                        r'\{.*\}', result_text, re.DOTALL
                    )
                    if json_obj_match:
                        json_str = json_obj_match.group(0)

                if not json_str:
                    logger.warning(
                        f"LLM node count extraction attempt {attempt}/{max_attempts}: "
                        f"no JSON found in response"
                    )
                    continue

                data = json.loads(json_str)
                unit_node_counts = data.get("unit_node_counts", {})

                if not unit_node_counts:
                    logger.warning(
                        f"LLM node count extraction attempt {attempt}/{max_attempts}: "
                        f"'unit_node_counts' key missing or empty"
                    )
                    continue

                # Validate
                result = {}
                all_valid = True
                for comp in type60_components:
                    unit_num = comp.get("unit_number", 0)
                    count = unit_node_counts.get(str(unit_num), None)

                    if (
                        count is not None
                        and isinstance(count, (int, float))
                        and int(count) > 0
                    ):
                        result[unit_num] = int(count)
                    else:
                        all_valid = False
                        logger.warning(
                            f"LLM node count extraction attempt {attempt}/{max_attempts}: "
                            f"UNIT {unit_num} got invalid count {count!r}"
                        )
                        break

                if all_valid and result:
                    logger.info(
                        f"LLM node count extraction succeeded on attempt {attempt}"
                    )
                    return result

            except (json.JSONDecodeError, ValueError, TypeError) as e:
                logger.warning(
                    f"LLM node count extraction attempt {attempt}/{max_attempts} "
                    f"parse error: {e}"
                )
            except Exception as e:
                logger.warning(
                    f"LLM node count extraction attempt {attempt}/{max_attempts} "
                    f"unexpected error: {e}"
                )

        # All attempts exhausted — fall back to 5 nodes
        logger.warning(
            f"LLM node count extraction failed after {max_attempts} attempts, "
            f"using default 5 nodes for Type60 Uniform variants"
        )
        return default_counts

