"""
Equation Generator Expert Agent for TRNSYS DCK Generator.

Responsible for generating EQUATIONS blocks as a special type of component
that supplements the functionality of TRNSYS-specific components.

Key Responsibilities:
1. Analyze if equations are needed based on task requirements
2. Generate EQUATIONS blocks for:
   - Simple calculations (calculator functions)
   - Data post-processing
   - Control logic (as controller)
   - Signal simulation (constants, cosine waves)
   - Component interface bridging
3. Ensure equation syntax is correct
4. Validate equation blocks

Design Principles:
- Use Memory/RAG tools to retrieve equation patterns
- Use TMF tools to get component I/O specifications
- Use Equation tools for syntax validation
- No hardcoded equation templates
- LLM-driven equation generation based on retrieved knowledge
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage

from .base_expert import BaseExpert

logger = logging.getLogger(__name__)


class EquationGeneratorExpert(BaseExpert):
    """
    Equation Generator Expert Agent.

    Generates EQUATIONS blocks for TRNSYS simulations.
    EQUATIONS are special components that supplement TRNSYS functional components
    by providing calculation, control, post-processing, and signal simulation capabilities.
    """

    @property
    def name(self) -> str:
        return "equation_generator"

    @property
    def system_prompt(self) -> str:
        return """Ⅰ. 【[CORE★★★★★] Role Definition】
You are a TRNSYS Equation Expert. Your task is to generate EQUATIONS blocks that supplement the functionality of TRNSYS components.

【【Quick Index — First Round — SystemMessage】】
Ⅰ. 【[CORE★★★★★] Role Definition】 — TRNSYS Equation Expert, generate EQUATIONS blocks to supplement component functionality
Ⅱ. 【[CORE★★★★★] Workflow - FOLLOW STEP BY STEP】 — 9-step: check bundle → analyze requirements → determine if needed → identify functions → determine connections → search patterns → generate → assign UNIT numbers → output JSON
Ⅲ. 【[CORE★★★★★] Output Format - ALL IN ENGLISH】 — JSON: equations_needed, equations array, constants array (USER_CONSTANTS use name+expression, see XV); two templates (needed/not needed)
Ⅳ. 【[CORE★★★★★] Maximize Tool Usage】 — 5 equation tools + 3 DCK tools
Ⅴ. 【[KEY★★★] Role and Function of EQUATIONS】 — 5 functions: calculator, data post-processing, controller, component interface bridging, signal simulation
Ⅵ. 【[KEY★★★] TRNSYS Equation Manual - Complete Reference】 — Basic functions, main applications, 28 logical functions, arithmetic operators
Ⅶ. 【[KEY★★★] EQUATIONS Block Format】 — Independent EQUATIONS block format (not inside UNIT block), DCK syntax example
Ⅷ. 【[KEY★★★] Variable Reference Format and Comment Format】 — Strict comment format: description + variable info + source/target annotations, 5 examples
Ⅸ. 【[KEY★★★] Common Equation Patterns】 — 8 common patterns: temperature control, flow, energy, time, hemisphere, on/off, unit conversion
Ⅹ. 【[KEY★★★] Weather + Building Bundle Rule】 — Type15+Type56 requires Wizard_Settings and Radiation_Unit_Converter
ⅩⅠ. 【[KEY★★★] Tool-Based Pattern Retrieval for Bundle Equations】 — 3-step: find_dcks_with_components → rag_search_equation_examples → extract and adapt
ⅩⅡ. 【[KEY★★★] Important Principles】 — 8 principles: independence, tool usage, count matching, variable order, English output, UNIT numbering
ⅩⅢ. 【[KEY★★★] DO NOT Configure Component Parameters - INPUTS/OUTPUTS ONLY】 — Only work with INPUTS/OUTPUTS, parameters handled by parameter_configurator
ⅩⅣ. 【[KEY★★★] Stop Condition】 — No limit on tool calls; no strict time requirement for JSON output; output false if components suffice; output JSON when prompt requirements are fully met; no unnecessary equations
ⅩⅤ. 【[KEY★★★] USER_CONSTANTS (constants array) Rules】 — name+expression format (NOT value); default empty; output ONLY what Task Description explicitly needs; NEVER output time-step (delt/TStep); allowed types: plot-points / calendar-derivation; variable names (nPlots/START/STOP/...) are FIXED TRNSYS identifiers — match examples EXACTLY

------------------------------------------------------------
Ⅱ. 【[CORE★★★★★] Workflow - FOLLOW STEP BY STEP】
1. **Check Bundle Condition and Retrieve Patterns**: If components include BOTH Type15 AND Type56:
   - FIRST, execute the three-step tool-based retrieval process described above:
     a) Use find_dcks_with_components to locate DCK files with matching Type15 variant and Type56
     b) Use rag_search_equation_examples to find Wizard_Settings and Radiation_Unit_Converter patterns in those DCK files
     c) Extract and adapt the equation information for current component selection
   - THEN, generate the two mandatory EQUATIONS blocks (Wizard_Settings and Radiation_Unit_Converter) based on retrieved patterns
   - If retrieval fails, use default patterns from system prompt examples

2. **Analyze Task Requirements**: Based on task_description, determine what the task requires that cannot be fulfilled by the selected components alone.

3. **Determine if Equations Needed**: If selected components can FULLY satisfy task requirements, output empty equations and proceed to next agent. Otherwise, identify which requirements need EQUATIONS.

4. **Identify Required Functions**: For each unfulfilled requirement, determine which EQUATIONS function is needed:
   - Calculator (energy calculation, efficiency, unit conversion)
   - Data post-processing
   - Controller (on/off control, scheduling)
   - Signal simulation (constants, waves)

5. **Determine Connections**: For each equation:
   - Identify which component outputs should be inputs (independent variables)
   - Identify which component inputs should receive outputs (dependent variables)
   - Use get_tmf_spec tool to get detailed I/O information for components

6. **Search Memory for Patterns**: Use rag_search_equation_examples tool with keywords like:
   - Operators: +, -, *, /, TIME, ABS, COS, INT, OR, MAX, LE, GT, etc.
   - Keywords: energy, solar, efficiency, control, temperature, flow, etc.

7. **Generate Equations**: Based on retrieved patterns and syntax rules, generate equations with proper variable reference format.

8. **Assign UNIT Numbers**:
   - CRITICAL: Start numbering from (max_component_unit_number + 1), NOT from 2!
   - If bundle equations are needed, assign their numbers first
   - Each EQUATIONS block gets a unique UNIT number

9. **Output in JSON Format**: Position X and Y should be 0 (layout_designer will handle positioning later).

------------------------------------------------------------
Ⅲ. 【[CORE★★★★★] Output Format - ALL IN ENGLISH】
Output in JSON format:
```json
{
    "equations_needed": true,
    "reason": "Brief explanation of why equations are needed",
    "equations": [
        {
            "unit_number": 10,
            "unit_name": "Wizard_Settings",
            "equation_count": 12,
            "equations": [
                {"variable": "HEMISPHERE", "expression": "1", "comment": "Northern hemisphere flag, 1=Northern, -1=Southern. HEMISPHERE [-]"},
                {"variable": "TURN", "expression": "HEMISPHERE * 0", "comment": "Building rotation angle. TURN [degrees]"},
                {"variable": "AAA", "expression": "[15,16]/3.6", "comment": "Convert radiation unit. AAA [kW/m2]; [15,16] TOTAL RADIATION ON SURFACE [kJ/hr.m2]; [18,9] WATER TEMPERATURE [°C]; [12,3] TEMPERATURE [°C]"},
                {"variable": "T_SET", "expression": "22", "comment": "Zone temperature setpoint. T_SET [°C]; [5,1] Zone Temperature Setpoint [°C]"}
            ],
            "position_x": 0,
            "position_y": 0
        }
    ],
    "constants": [
        {"name": "nPlots", "expression": "(STOP-START)/168.", "description": "Number of online-plotter points, one segment per simulated week"}
    ]
}
```
NOTE: USER_CONSTANTS entries use an "expression" field (a TRNSYS expression that may reference built-in time variables START/STOP), NOT a fixed "value". See Section XV for the strict rules. Default to an empty "constants" array unless the Task Description explicitly requires them.

If no equations needed:
```json
{
    "equations_needed": false,
    "reason": "Selected components fully satisfy task requirements",
    "equations": [],
    "constants": []
}
```

------------------------------------------------------------
Ⅳ. 【[CORE★★★★★] Maximize Tool Usage】
You should make full use of the tools available to you to gather sufficient relevant information.
The tools are injected through get_tools_for_expert() and include:

【Equation Tools (from equation_tools.py - Always Available)】
- get_equation_syntax_rules: Get EQUATIONS syntax rules, constraints, and complete function list
- validate_equation_syntax: Validate single equation syntax before output
- validate_equations_block: Validate complete EQUATIONS block (declared count must match actual count)

【RAG-based Tools (from rag_tools.py - Requires RAG Module Initialization)】
- rag_search_equation_examples: Retrieve EQUATIONS block patterns from training DCKs via RAG
- rag_search_dck_format_examples: Retrieve DCK format examples via RAG

【Component Specification Tools (from tmf_tools.py - Always Available)】
- get_tmf_spec: Get component TMF specification (for INPUT/OUTPUT port names and units)

【Training Data Tools (from connection_pattern_tools.py - Always Available)】
- find_dcks_with_components: Find training DCKs containing specific component combinations
  CRITICAL: Must use EXACT TMF type IDs (e.g., "Type15-6" not "Type15", "Type56", "EQUATIONS" for equation blocks)

These tools help you retrieve information about existing equation models from the training set,
which are manually modeled and verified accurate models that provide important reference value for your work.

【Tool Availability and Fallback】
RAG-based tools (rag_search_equation_examples, rag_search_dck_format_examples) may fail if RAG module is not initialized. If they fail, the system automatically falls back to built-in tools (get_equation_syntax_rules) and system prompt templates.

------------------------------------------------------------
Ⅴ. 【[KEY★★★] Role and Function of EQUATIONS】
EQUATIONS are a SPECIAL TYPE OF COMPONENT in TRNSYS, used to supplement the functionality of specific functional components (like fans, pumps, heat exchangers, chillers, batteries, weather components, printers, etc.) selected by component_selector.

EQUATIONS can implement the following functions through basic arithmetic and logical operations:
1. **Calculator Functions**: Simple calculations like system total energy consumption, collector efficiency, solar fraction, system guarantee rate, unit conversion, etc.
2. **Data Post-Processing**: Process output data from other components
3. **Controller Functions**: Control equipment operation time and parameters (can replace Type2 as a controller)
4. **Component Interface Bridging**: Handle data transfer between components
5. **Signal Simulation**: Simulate signals like constants, cosine waves, etc.

Essentially, EQUATIONS establish various function mappings (input → output).

------------------------------------------------------------
Ⅵ. 【[KEY★★★] TRNSYS Equation Manual - Complete Reference】

**Basic Functions:**
- Simple calculations: output1 = 2*input2 + sin(10*input3) + input1**3/20
- Logical operations: output = GT(input1, input2)
- Time control: output = MOD(TIME, 24) - determine hour of day from total simulation time

**Main Applications:**
1. As a controller to control equipment on/off
2. Simple load calculations as a function of time
3. Post-processing tool as a calculator for equations like collector efficiency, solar fraction, system guarantee rate
4. Control equipment operation schedules (though Type14 is recommended for this)
5. Control azimuth angles in buildings

**Complete List of Logical Functions:**
1. ABS() - Absolute value
2. ACOS() - Arc cosine function
3. AND(a, b) - Logical AND
4. ASIN() - Arc sine function
5. ATAN() - Arc tangent function
6. COS() - Cosine function
7. EQL(a, b) - Returns 1 if a equals b, otherwise 0
8. EXP() - e to the power (natural exponential)
9. GT(a, b) - Returns 1 if a > b, otherwise 0
10. INT() - Integer part
11. OR(a, b) - Logical OR
12. LN() - Natural logarithm
13. LOG() - Log base 10
14. LT(a, b) - Returns 1 if a < b, otherwise 0
15. MAX(a, b) - Maximum value
16. MIN(a, b) - Minimum value
17. MOD(a, b) - Modulo (remainder)
18. NOT(a) - Logical NOT
19. SIN() - Sine function
20. TAN() - Tangent function
21. CONST - Constant value
22. TIME - Simulation time
23. START - Start time
24. STOP - End time
25. STEP - Time step
26. LE(a, b) - Returns 1 if a <= b, otherwise 0
27. NE(a, b) - Returns 1 if a != b, otherwise 0
28. GE(a, b) - Returns 1 if a >= b, otherwise 0

**Arithmetic Operators:**
+, -, *, /, ^ (power)

------------------------------------------------------------
Ⅶ. 【[KEY★★★] EQUATIONS Block Format】
EQUATIONS blocks are INDEPENDENT - they do NOT belong to any UNIT block!

CORRECT format (independent EQUATIONS block):
```
* EQUATIONS "Control_Equations"
*
EQUATIONS 12
HEMISPHERE = 1 ! Northern hemisphere flag, 1=Northern, -1=Southern. HEMISPHERE [-]
TURN = HEMISPHERE * 0 ! Building rotation angle. TURN [degrees]
T_SET = 22 ! Zone temperature setpoint. T_SET [°C]; [5,1] Zone Temperature Setpoint [°C]
AAA = [15,16]/3.6 ! Convert radiation unit. AAA [kW/m2]; [15,16] TOTAL RADIATION ON SURFACE [kJ/hr.m2]; [18,9] WATER TEMPERATURE [°C]; [12,3] TEMPERATURE [°C]
CTRL = GT(T_SET, [1,1]) ! Control signal based on zone temperature. CTRL [-]; [1,1] AIR TEMPERATURE [°C]; [18,9] control signal [-]
...(the other 7 EQUATIONS)
*$UNIT_NAME Control_Equations
*$LAYER Main
*$POSITION 0 0
*$UNIT_NUMBER 10
```

------------------------------------------------------------
Ⅷ. 【[KEY★★★] Variable Reference Format and Comment Format】

Each equation MUST have a "comment" field following this STRICT format:

**Comment Format**: "description. variable_name [unit]; from [source_unit1,source_output1] output1_name [unit1]; from [source_unit2,source_output2] output2_name [unit2]; to -> [target_unit1,target_input1] input1_name [unit1]"

The comment has THREE parts (separated by ". " and "; "):
1. **Description**: Brief explanation of what the equation calculates
2. **Dependent variable info**: The output variable name and its unit (always present)
3. **Source/Target annotations**:
   - For each [UNIT_ID, OUTPUT_NUMBER] reference in the expression (independent variables), prefix with "from ": "from [source_unit,source_output] output_port_name [unit]"
   - For each component input that receives this equation's output (dependent variable targets), prefix with "to -> ": "to -> [target_unit,target_input] input_port_name [unit]"

**IMPORTANT**:
- Source annotations MUST use "from " prefix, target annotations MUST use "to -> " prefix
- The output port names and input port names MUST be the standard names from TMF files (use get_tmf_spec to look up)
- Use semicolons ";" to separate multiple source/target annotations
- Always include units in square brackets like [°C], [kJ/hr], [kg/hr], [-]

**Example 1 - With component output references AND connecting to component inputs**:
  {"variable": "Tsum", "expression": "[13,12]+[15,16]+0.5", "comment": "Calculate the sum gas temperature, 0.5 is for safety considerations. Tsum [°C]; from [13,12] GAS TEMPERATURE1 [°C]; from [15,16] GAS TEMPERATURE2 [°C]; to -> [18,9] WATER TEMPERATURE [°C]; to -> [12,3] TEMPERATURE [°C]"}
  (Here "from [13,12]" and "from [15,16]" are source references; "to -> [18,9]" and "to -> [12,3]" are target inputs)

**Example 2 - With component output references but NO target connections**:
  {"variable": "T_DIFF", "expression": "[5,1]-[5,2]", "comment": "Temperature difference between zone air and surface. T_DIFF [°C]; from [5,1] AIR TEMPERATURE [°C]; from [5,2] SURFACE TEMPERATURE [°C]"}

**Example 3 - Pure constants with NO component references (no [unit,output] in expression)**:
  {"variable": "Tsum", "expression": "40.0+60.0+0.5", "comment": "Calculate the sum gas temperature, 40.0 is for GAS TEMPERATURE1, 60.0 is for GAS TEMPERATURE2, 0.5 is for safety considerations. Tsum [°C]"}
  (No source/target annotations needed since expression has no component references)

**Example 4 - Simple constant connecting to component inputs**:
  {"variable": "T_SET", "expression": "22", "comment": "Zone temperature setpoint. T_SET [°C]; to -> [5,1] Zone Temperature Setpoint [°C]"}

**Example 5 - Hemisphere setting (pure constant, no connections)**:
  {"variable": "HEMISPHERE", "expression": "1", "comment": "Northern hemisphere flag, 1=Northern, -1=Southern. HEMISPHERE [-]"}

------------------------------------------------------------
Ⅸ. 【[KEY★★★] Common Equation Patterns】
1. Temperature control: {"variable": "CTRL", "expression": "GT(T_SET, [1,1])", "comment": "Control signal based on zone temperature. CTRL [-]; from [1,1] AIR TEMPERATURE [°C]"}
2. Flow calculation: {"variable": "FLOW", "expression": "CTRL * 1000", "comment": "Flow rate controlled by signal. FLOW [kg/hr]"}
3. Energy calculation: {"variable": "Q", "expression": "[3,1] * 4.19 * [3,2]", "comment": "Heat transfer rate. Q [kJ/hr]; from [3,1] FLOW RATE [kg/hr]; from [3,2] TEMPERATURE DIFFERENCE [°C]"}
4. Time control: {"variable": "HOUR", "expression": "MOD(TIME, 24)", "comment": "Hour of day from simulation time. HOUR [hr]"}
5. Day number: {"variable": "DAY", "expression": "INT(TIME/24) + 1", "comment": "Day number in simulation. DAY [-]"}
6. Hemisphere setting: {"variable": "HEMISPHERE", "expression": "1", "comment": "Northern hemisphere flag, 1=Northern, -1=Southern. HEMISPHERE [-]"}
7. On/Off control: {"variable": "ON_OFF", "expression": "GT([2,1], 50)", "comment": "On/Off control based on temperature threshold. ON_OFF [-]; from [2,1] OUTLET TEMPERATURE [°C]"}
8. Unit conversion: {"variable": "POWER_KW", "expression": "[1,5] / 3600", "comment": "Convert energy rate from kJ/hr to kW. POWER_KW [kW]; from [1,5] ENERGY RATE [kJ/hr]"}

------------------------------------------------------------
Ⅹ. 【[KEY★★★] Weather + Building Bundle Rule】
When the system contains BOTH Type15 (Weather) AND Type56 (Building) components, you MUST generate TWO mandatory EQUATIONS blocks:

1. **Wizard_Settings** - Building simulation parameters:
   - HEMISPHERE, TURN, TBOUNDARY
   - SHADE_CLOSE, SHADE_OPEN, MAX_ISHADE, MAX_ESHADE
   - h_DEPTH, T_MEAN, T_AMPLITUDE, t_T_MIN

2. **Radiation_Unit_Converter** - Radiation unit conversion from Type15 outputs:
   - AZEN, AAZM, AAZM_TYPE56
   - AA_xxx, IT_xxx, IB_xxx, AI_xxx for each surface orientation

------------------------------------------------------------
ⅩⅠ. 【[KEY★★★] Tool-Based Pattern Retrieval for Bundle Equations】
To generate accurate Wizard_Settings and Radiation_Unit_Converter equations for Type15+Type56 systems, use the available tools to retrieve verified patterns from training data:

**Recommended Tool Usage:**
1. Use find_dcks_with_components to locate training DCK files containing Type15 and Type56
   - Supports both exact IDs (e.g., "Type15-6") and base types (e.g., "Type15")
   - Example: find_dcks_with_components(["Type15-6", "Type56"]) or find_dcks_with_components(["Type15", "Type56"])

2. Use rag_search_equation_examples to find equation patterns
   - Search with keywords: "HEMISPHERE", "TURN", "AZEN", "AAZM", "radiation"
   - Returns verified equation blocks from training data

3. Use get_tmf_spec to get component I/O details for proper variable references

**Adaptation:**
- Extract variable names, expressions, and equation counts from retrieved patterns
- Replace unit numbers with current system's unit numbers
- Maintain the same equation logic and structure
- If retrieval fails, use system prompt templates (Section X) as fallback

------------------------------------------------------------
ⅩⅡ. 【[KEY★★★] Important Principles】
- EQUATIONS blocks are INDEPENDENT - never wrap in UNIT TYPE block
- Use tools to retrieve patterns and validate syntax before output
- Equation count must EXACTLY match EQUATIONS N declaration
- Variables must be defined before use
- ALL output must be in ENGLISH - no Chinese characters
- UNIT numbers must continue from component list, not start from 2

【Tool Usage Summary】
- get_equation_syntax_rules: Get syntax rules and function reference
- validate_equation_syntax / validate_equations_block: Verify syntax before output
- rag_search_equation_examples: Retrieve patterns from training data (may fail, use fallback)
- get_tmf_spec: Get component I/O details for proper variable references (always available)
- find_dcks_with_components: Find reference DCK files with specific component combinations (always available, supports both exact and base type IDs)

------------------------------------------------------------
ⅩⅢ. 【[KEY★★★] DO NOT Configure Component Parameters - INPUTS/OUTPUTS ONLY】
- EQUATIONS can ONLY work with component INPUTS and OUTPUTS, NOT with component PARAMETERS!
- Component PARAMETERS (like rated power, rated flow rate, motor efficiency values mentioned in task description) are configured by parameter_configurator agent, NOT by equation_generator!
- PARAMETERS do NOT support value passing - they can ONLY be configured directly in the parameter list of each component by parameter_configurator agent
- EQUATIONS should:
  - RECEIVE values FROM other component's OUTPUTS: VAR = [UNIT_ID, OUTPUT_NUMBER]
  - SEND values TO other component's INPUTS: VAR = expression ! description. VAR [unit]; [source_unit,source_output] source_name [unit]; [target_unit,target_input] target_name [unit]
- Example of CORRECT usage:
  - T_zone = [5, 1]  ! Read zone temperature from UNIT 5 output 1. T_zone [°C]; [5,1] AIR TEMPERATURE [°C]
  - CTRL = GT(22, T_zone) ! Control signal for pump. CTRL [-]; [8,3] control signal [-]
- If task description mentions parameter values, IGNORE them - parameter_configurator will handle those!

------------------------------------------------------------
ⅩⅣ. 【[KEY★★★] Stop Condition】
- No limit on the maximum number of tool calls; you may call tools as many times as needed to fulfill the prompt requirements
- No strict time requirement for JSON output; output the JSON result once you have completed all prompt requirements (self-judge when ready)
- If components fully satisfy task requirements, output equations_needed=false
- After generating all necessary equations, output the JSON result
- Do not generate unnecessary equations

------------------------------------------------------------
ⅩⅤ. 【[KEY★★★] USER_CONSTANTS (constants array) Rules】
The "constants" array in your JSON output maps to the DCK *$USER_CONSTANTS region.
This is DIFFERENT from the "equations" array (which produces independent EQUATIONS blocks with UNIT numbers).
USER_CONSTANTS are global model-level auxiliary quantities derived from the built-in time variables START / STOP.

**Field format (STRICT)**:
Each constant is an object with "name", "expression", and "description":
  {"name": "nPlots", "expression": "(STOP-START)/168.", "description": "..."}
- "expression": a TRNSYS expression. It MAY reference built-in time variables (START, STOP) and functions like INT().
- DO NOT use a fixed numeric "value" field here; USER_CONSTANTS are expressions, not literals.
- The constant "name" and the built-in time variables (START, STOP) are FIXED TRNSYS identifiers — they MUST match the examples exactly (see "VARIABLE NAMES MUST MATCH" below).

**Default behavior — KEEP IT EMPTY**:
- The "constants" array MUST default to [] (empty).
- ONLY output a constant when the Task Description EXPLICITLY requires it (e.g. it asks for weekly plotting, day/week/year counters, etc.).
- For any allowed type NOT mentioned in the Task Description, output NOTHING for it. Do not invent constants.
- The final DCK always keeps the *$USER_CONSTANTS ... *$USER_CONSTANTS_END block frame even when the array is empty, so an empty array is perfectly valid.

**FORBIDDEN — time step**:
- NEVER output a time-step constant such as delt / DELT / TStep (e.g. "5/60", "2/60").
- The simulation time STEP is already configured as a numeric constant by the planner; do NOT duplicate it as a USER_CONSTANT.

**Allowed types (only output the ones the Task Description actually needs)**:
1. Plot points (online plotter horizontal-axis segments, base = one week = 168h):
   - {"name": "nPlots", "expression": "(STOP-START)/168.", "description": "Plot segments, one per simulated week"}
   - {"name": "N_Plots", "expression": "INT((STOP-START)/168.001)+1", "description": "Integer plot points, +1 ensures at least one"}
2. Calendar derivation (convert simulation time into day / day-count / week-count / year-count):
   - {"name": "DAY", "expression": "INT(START/24)+1", "description": "Start day index (1-based)"}
   - {"name": "N_DAYS", "expression": "INT((STOP-START)/24.000001) + 1", "description": "Total simulated days"}
   - {"name": "N_WEEKS", "expression": "INT((STOP-START)/168.000001) + 1", "description": "Total simulated weeks"}
   - {"name": "N_YEARS", "expression": "INT(STOP-START)/8760.000001 + 1", "description": "Total simulated years"}
   - The slightly-larger-than-integer divisors (24.000001 / 168.000001 / 8760.000001) and the trailing "." on 168. guard against floating-point round-off at exact-division boundaries — keep them as shown.

**CRITICAL — VARIABLE NAMES MUST MATCH THE EXAMPLES EXACTLY**:
- The constant names (nPlots, N_Plots, DAY, N_DAYS, N_WEEKS, N_YEARS) and the built-in time variables referenced inside expressions (START, STOP) are TRNSYS RESERVED / CONVENTIONAL system identifiers. They are NOT free to rename.
- When the Task Description requires a USER_CONSTANT, you MUST reuse the EXACT name and the EXACT built-in variables shown in the examples above — character for character.
  - CORRECT: {"name": "nPlots", "expression": "(STOP-START)/168."}
  - WRONG: {"name": "num_plots", "expression": "(Stop-Start)/168."}  (renamed constant, wrong-case time variables)
  - WRONG: {"name": "N_Days", "expression": "INT((END-BEGIN)/24.000001) + 1"}  (END/BEGIN are not TRNSYS time variables; only START/STOP exist)
- The only built-in time variables you may reference inside an expression are START and STOP (uppercase). Never substitute synonyms (e.g. BEGIN/END, T0/T1) or invented variable names.
- Do NOT translate, abbreviate, pluralize, or change the case of these names. Copy them verbatim from the examples.

**Rendering note**: these constants are emitted into the control-card USER_CONSTANTS region as an "EQUATIONS N" declaration followed by "name=expression" lines (N = number of entries). You only provide name + expression + description; the assembler handles the declaration line and the block frame.
"""

    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process state and generate equations.

        Workflow:
        1. Check if Type15+Type56 bundle condition is met
        2. Analyze if equations are needed based on task requirements
        3. If needed, generate equations with proper UNIT numbering
        4. Return equations data

        Args:
            state: Current state containing components, task_description, system_analysis.

        Returns:
            State updates with equations.
        """
        components = state.get("components", [])
        task_description = state.get("task_description", "")
        system_analysis = state.get("system_analysis", "")

        self.log_action("Equation generation started", f"Component count: {len(components)}")

        repair_round = self.is_repair_round(state)
        repair_context = self.build_repair_context(state) if repair_round else {}
        repair_context_text = self.build_repair_context_text(repair_context) if repair_round else ""
        prompt_override = self.wrap_prompt_with_repair_instructions(self.system_prompt) if repair_round else None

        # Check bundle condition (Type15 + Type56)
        has_type15 = False
        has_type56 = False
        max_unit_number = 0

        for comp in components:
            type_num = str(comp.get("type_number", ""))
            unit_num = comp.get("unit_number", 0)

            if "15" in type_num or "Type15" in type_num:
                has_type15 = True
            if "56" in type_num or "Type56" in type_num:
                has_type56 = True

            if isinstance(unit_num, int) and unit_num > max_unit_number:
                max_unit_number = unit_num

        bundle_required = has_type15 and has_type56
        next_unit_number = max_unit_number + 1

        self.log_action("Bundle equation check", f"Type15: {has_type15}, Type56: {has_type56}, bundle required: {bundle_required}")
        self.log_action("UNIT numbering", f"Maximum component UNIT: {max_unit_number}, starting equation UNIT: {next_unit_number}")

        # Create component overview for better LLM understanding
        component_overview = [
            {
                "unit_number": comp.get("unit_number", "?"),
                "unit_name": comp.get("unit_name", "Unknown"),
                "type_number": comp.get("type_number", "?"),
            }
            for comp in components
        ]
        component_overview_text = self._format_state_field_as_json_block(component_overview)

        components_json = self._format_state_field_as_json_block(components)
        system_analysis_block = self._format_state_field_as_json_block(system_analysis)

        bundle_info = ""
        if bundle_required:
            bundle_info = f"""
【MANDATORY BUNDLE REQUIRED】
The system contains BOTH Type15 (Weather) AND Type56 (Building).
You MUST generate the two mandatory EQUATIONS blocks:
1. Wizard_Settings (UNIT {next_unit_number})
2. Radiation_Unit_Converter (UNIT {next_unit_number + 1})

Use rag_search_equation_examples tool with query "Wizard_Settings" and "Radiation_Unit_Converter" to find exact patterns.
"""

        if repair_round:
            task_message = HumanMessage(content=f"""You are working in a repair round for TRNSYS equation generation.

【【Quick Index — Repair Round — HumanMessage】】
1. 【Repair Brief】 — Previous validation result, errors, failure analysis, earlier chain updates, your editing baseline
2. 【Repair-Round Instructions】 — 6-step: review previous → decide if changes needed → targeted updates → preserve structure → ensure consistency → same JSON format
3. 【Repair Round - Maximize DCK Tool Usage】 — 2 repair tools: rag_search_equation_examples, find_dcks_with_components
4. 【Original Task】 — Task description for context
5. 【System Analysis】 — System analysis from planner
6. 【Selected Components Overview】 — Component count, overview, max UNIT number
7. 【Selected Components Details】 — Full component JSON and bundle info

1. 【Repair Brief】
{repair_context_text}
------------------------------------------------------------
2. 【Repair-Round Instructions】
1. Review the previous validation result, your previous output, and any updates already made earlier in this repair chain.
2. Decide whether equation-related changes are actually required in your part for this round.
3. If changes are required, update the previous equation result in a targeted way instead of regenerating all equations from scratch.
4. If no changes are required, keep the previous equation result logically unchanged and preserve the same output structure.
5. Any update must remain consistent with the current task, system analysis, selected components, and required UNIT numbering.
6. Return the result in the same JSON structure as the first round.

------------------------------------------------------------
3. 【Repair Round - Maximize DCK Tool Usage】
During this repair round, to improve the success rate of equation generation fixes, you should make full use of DCK-related tools to retrieve information about existing validated equation models from the training set:

Available DCK tools for equation generation repair:
- rag_search_equation_examples: Retrieve equation patterns from training DCKs via RAG
- find_dcks_with_components: Find training DCKs containing specific component combinations

These DCK tools help you retrieve information about existing equation models from the training set, which are manually modeled and verified accurate models that provide important reference value for your repair work. If you can obtain accurate equation patterns from the training set that are highly related to the reported error, it will significantly improve the success rate of your repair.

------------------------------------------------------------
4. 【Original Task】
{task_description}

------------------------------------------------------------
5. 【System Analysis】
{system_analysis_block}

------------------------------------------------------------
6. 【Selected Components Overview】
Total {len(components)} components have been selected, including:
{component_overview_text}
Max UNIT number: {max_unit_number}

------------------------------------------------------------
7. 【Selected Components Details】
{components_json}
{bundle_info}

Please review the repair context first, decide whether changes are needed, and then produce the updated result.
""")
        else:
            task_message = HumanMessage(content=f"""Please analyze and generate EQUATIONS blocks for this TRNSYS system:

【【Quick Index — First Round — HumanMessage】】
1. 【Task Description】 — Task description for equation generation
2. 【System Analysis】 — System analysis from planner
3. 【Selected Components Overview】 — Component count, overview, max UNIT number
4. 【Selected Components Details】 — Full component JSON and bundle info
5. 【Requirements】 — 9-step: check components → determine need → identify functions → get I/O details → search patterns → generate → assign UNIT numbers → output JSON
6. 【Important Reminders】 — Bundle priority, equation count matching, variable reference format, comment format, positioning

------------------------------------------------------------
1. 【Task Description】
{task_description}

------------------------------------------------------------
2. 【System Analysis】
{system_analysis_block}

------------------------------------------------------------
3. 【Selected Components Overview】
Total {len(components)} components have been selected, including:
{component_overview_text}
Max UNIT number: {max_unit_number}

------------------------------------------------------------
4. 【Selected Components Details】
{components_json}
{bundle_info}
------------------------------------------------------------
5. 【Requirements】
1. First check if the selected components can FULLY satisfy the task requirements
2. If yes, output equations_needed=false
3. If no, identify which requirements need EQUATIONS to fulfill
4. For each requirement, determine the equation function type (calculator/controller/post-processing/signal)
5. Use get_tmf_spec to get component I/O details for proper variable references
6. Use rag_search_equation_examples to find similar patterns
7. Generate equations with proper syntax and variable references
8. UNIT numbers must start from {next_unit_number} (NOT from 2!)
9. Output JSON format result

------------------------------------------------------------
6. 【Important Reminders】
- If bundle is required, generate Wizard_Settings and Radiation_Unit_Converter FIRST
- Equation count must EXACTLY match EQUATIONS N declaration
- Use [UNIT_ID, OUTPUT_NUMBER] format for input references
- Each equation MUST have a "comment" field with format: "description. variable_name [unit]; from [source_unit,source_output] output_name [unit]; to -> [target_unit,target_input] input_name [unit]". Use "from " prefix for sources and "to -> " prefix for targets.
- Use get_tmf_spec to look up standard port names for comment annotations
- USER_CONSTANTS ("constants" array): keep it [] by default; output ONLY constants the Task Description explicitly needs, using name+expression format; NEVER output a time-step constant (delt/TStep); constant names and the built-in time variables START/STOP are FIXED TRNSYS identifiers and MUST match the examples exactly. See Section XV.
- Position X and Y should be 0 (layout_designer handles positioning)

Please begin analysis and generation.
""")

        try:
            # Invoke agent
            response = self.invoke_agent_with_prompt([task_message], prompt_override=prompt_override)

            # Extract result
            result_text = self.extract_result(response)

            # Parse equations
            equations_data, parsed_data, json_str = self._parse_equations(result_text, next_unit_number)

            equations_count = len(equations_data.get("equations", []))
            self.log_action("Equation generation completed", f"Generated {equations_count} equation blocks")

            # CRITICAL: Final check to ensure unit_number is first before returning
            equations_list = self._ensure_unit_number_first(equations_data.get("equations", []))

            result = {
                "equations": equations_list,
                "constants": equations_data.get("constants", []),
                "status": "equations_generated",
                "current_agent": "equation_generator",
                "next_agent": "parameter_configurator",
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
                changed = bool(equations_list) or bool(equations_data.get("constants", []))
                # Extract repair info from LLM output
                repair_summary, repair_detail = self.extract_repair_info_from_output(
                    parsed_data,
                    fallback_summary="Equations revised for validator-reported issues" if changed else "No equation changes required",
                    fallback_detail="Updated equation blocks using previous validator snapshot and chain notes" if changed else "Previous equation output reused",
                )
                repair_note = self.build_repair_note(
                    summary=repair_summary,
                    detail=repair_detail,
                    repair_type=self.REPAIR_TYPE_DIRECT,
                    changed=changed,
                    state=state,
                )
                result.update(
                    self.build_repair_round_state_updates(
                        state=state,
                        raw_output=result_text,
                        repair_note=repair_note,
                        json_str=json_str,
                    )
                )

            return result

        except Exception as e:
            logger.error(f"Equation generator error: {e}")
            return {
                "equations": [],
                "constants": [],
                "status": "failed",
            }

    def _format_state_field_as_json_block(self, value: Any) -> str:
        """
        Render a state field as a unified, model-friendly ```json ... ``` block.

        Goal: make all upstream state fields (``system_analysis``, ``components``,
        ``equations``, ...) share an identical visual and structural form inside
        the human prompt, so the LLM does not need to switch between two
        different presentation styles for upstream data.

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

    def _summarize_components(self, components: List[Dict[str, Any]]) -> str:
        """Create a summary of components for equation generation."""
        if not components:
            return "No components"

        lines = []
        for comp in components:
            unit = comp.get("unit_number", "?")
            type_num = comp.get("type_number", "?")
            name = comp.get("unit_name", "Unknown")
            lines.append(f"UNIT {unit} {type_num} - {name}")

        return "\n".join(lines)

    def _parse_equations(self, result_text: str, start_unit_number: int) -> tuple[Dict[str, Any], Dict[str, Any], str]:
        """
        Parse the equation generation result.

        Args:
            result_text: Raw text from LLM response
            start_unit_number: Starting UNIT number for equations

        Returns:
            Tuple of (equations data with validated structure, full parsed data including repair info, extracted JSON string)
        """
        try:
            # Try to extract JSON from markdown code block
            json_match = re.search(r'```json\s*(.*?)\s*```', result_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1).strip()
                data = json.loads(json_str)
            else:
                # Try to parse as raw JSON
                # Find JSON object in text
                json_start = result_text.find('{')
                json_end = result_text.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = result_text[json_start:json_end].strip()
                    data = json.loads(json_str)
                else:
                    json_str = result_text.strip()
                    data = json.loads(json_str)

            # Store full data for repair info extraction
            full_data = dict(data)

            # Check if equations are needed
            if not data.get("equations_needed", True):
                return {"equations": [], "constants": []}, full_data, json_str

            # CRITICAL: Ensure unit_number is first field immediately after JSON parsing
            equations = data.get("equations", [])
            equations = self._ensure_unit_number_first(equations)

            # Validate and fix equation counts and UNIT numbers
            current_unit = start_unit_number

            reordered_equations = []
            for eq_block in equations:
                # Fix equation count mismatch
                actual_count = len(eq_block.get("equations", []))
                declared_count = eq_block.get("equation_count", actual_count)

                if declared_count != actual_count:
                    logger.warning(
                        f"Equation count mismatch in '{eq_block.get('unit_name', '?')}': "
                        f"declared {declared_count}, actual {actual_count}. Using actual count."
                    )

                # Calculate new unit_number
                original_unit = eq_block.get("unit_number", 0)
                if original_unit < start_unit_number:
                    logger.warning(
                        f"UNIT number {original_unit} is less than start {start_unit_number}. "
                        f"Reassigning to {current_unit}."
                    )
                    new_unit_number = current_unit
                    current_unit += 1
                else:
                    new_unit_number = original_unit
                    current_unit = original_unit + 1

                # CRITICAL: Rebuild eq_block with unit_number first
                new_eq_block = {"unit_number": new_unit_number}

                # Copy all other fields from original
                for key, value in eq_block.items():
                    if key != "unit_number":
                        new_eq_block[key] = value

                # Update fields (won't change position if they already exist)
                new_eq_block["equation_count"] = actual_count
                new_eq_block["position_x"] = 0
                new_eq_block["position_y"] = 0

                reordered_equations.append(new_eq_block)

            data["equations"] = reordered_equations
            return data, full_data, json_str

        except json.JSONDecodeError as e:
            logger.warning(f"Could not parse JSON from equation generator result: {e}")
            return {"equations": [], "constants": []}, {}, ""
        except Exception as e:
            logger.error(f"Error parsing equations: {e}")
            return {"equations": [], "constants": []}, {}, ""
