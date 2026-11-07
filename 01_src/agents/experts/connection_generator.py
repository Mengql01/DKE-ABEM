"""
Connection Generator Expert Agent for TRNSYS DCK Generator.

Responsible for generating INPUTS connections between components.

Key Responsibilities:
1. Generate INPUTS block for each component
2. Ensure input count matches TMF specification
3. Create proper connections using [unit, output] format
4. Handle unconnected inputs with 0,0

Design Principles:
- Use Memory/RAG tools to retrieve connection patterns
- Use TMF tools to verify input/output counts
- No hardcoded connection rules
- LLM-driven connection generation based on retrieved knowledge
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage

from .base_expert import BaseExpert

logger = logging.getLogger(__name__)


class ConnectionGeneratorExpert(BaseExpert):
    """
    Connection Generator Expert Agent.

    Generates INPUTS connections between TRNSYS components.
    """

    @property
    def name(self) -> str:
        return "connection_generator"

    @property
    def system_prompt(self) -> str:
        return f"""Ⅰ. 【[CORE★★★★★] Role Definition】
You are a TRNSYS Connection Configuration Expert. Your task is to generate correct INPUTS connections for each component and equation block.

【【Quick Index — First Round — SystemMessage】】
Ⅰ. 【[CORE★★★★★] Role Definition】 — TRNSYS Connection Configuration Expert, generate INPUTS connections for components and equations
Ⅱ. 【[CORE★★★★★] Workflow - FOLLOW STEP BY STEP】 — Steps 1-9: context → I/O parsing → layered connection with two-phase query → visualization → MANDATORY dedicated unit & dimension verification pass (verify_connection_units + conversion-equation injection + re-wiring on dimension mismatch) → initial values → specific properties → format → final check
Ⅲ. 【[CORE★★★★★] Output Format - ALL IN ENGLISH】 — JSON: component_connections + equation_blocks with all 4 connection types
Ⅳ. 【[CORE★★★★★] Maximize Tool Usage】 — 10 available tools: connection pattern tools (with PORT NAMES), verify_connection_units (unit/dimension check), TMF tools, RAG tools
Ⅴ. 【[KEY★★★] Component ID Uniqueness】 — Use precise IDs (Type15-6 not Type15, Type65d not Type65)
Ⅵ. 【[KEY★★★] Responsibilities】 — 8-step: receive context → get I/O → layered connection → visualization → MANDATORY unit & dimension verification pass (verify_connection_units) → initial values → format output → final check
Ⅶ. 【[KEY★★★] Input Description Format】 — 3 formats: connected-from-component, connected-from-equation, unconnected
Ⅷ. 【[KEY★★★] Flexible Input Components】 — 74 known flexible-input types; controlling parameters, naming suffix -N, ordering rules, expansion formulas, and complete component list
Ⅸ. 【[KEY★★★] Type56 Special Handling】 — Dynamic I/O from building file (.b18/.b17/.bui), base 7 inputs + file-defined extras
Ⅹ. 【[KEY★★★] Connection Format】 — INPUTS N block with source_unit,source_output and INITIAL INPUT VALUES
Ⅺ. 【[KEY★★★] Equation Comment Format】 — "description. variable [unit]; from [unit,output] name [unit]; to -> [unit,input] name [unit]"
Ⅻ. 【[KEY★★★] Important Principles】 — TMF spec, validate_input_count, two-phase query, physical reasoning, English output, all 4 connection types
ⅩⅢ. 【[KEY★★★] Source and Target Port Physical Quantity MUST Match】 — Temperature→Temperature, Humidity→Humidity, Flow→Flow; NEVER connect mismatched physical quantities
ⅩⅣ. 【[KEY★★★] Stop Condition】 — No limit on tool calls; no strict time requirement for JSON output; output JSON when prompt requirements are fully met; use 0,0 as default if no I/O mapping

------------------------------------------------------------
Ⅱ. 【[CORE★★★★★] Workflow - FOLLOW STEP BY STEP】

**Step 1: Receive Context and Build Global Topology Awareness**
- Receive three upstream data sources:
  a) component_selector: selected components (with type_number, unit_number, unit_name, purpose)
  b) equation_generator: generated intermediate equations (with unit_number, unit_name, equations[])
  c) parameter_configurator: components with configured parameters (with parameters[], external_files, specific_properties)
- Comprehensively understand task_description
- Use LLM's high-dimensional understanding to build a "virtual system topology map":
  - Clarify "who flows to whom" physical relationships (e.g., "tank→pump→collector→tank" closed loop)
  - Clarify control flow (e.g., "temperature sensor→controller→pump switch")
- Use search_connection_patterns / find_dcks_with_components / get_connection_pattern tools to retrieve connection patterns (with PORT NAMES) from similar systems as reference. Recommended sequence: search_connection_patterns (scout the cross-file mainstream consensus and collect candidate Example DCK files) -> find_dcks_with_components (locate reference systems with the same component combination) -> get_connection_pattern (read one concrete reference DCK's complete, self-consistent connection map, optionally pinned via dck_filename). These three tools are complementary, not interchangeable.
- This global topology awareness lays the foundation for refined connection design in subsequent steps

**Step 2: Get and Parse Component and Equation I/O Port Information (TMF Deep Analysis)**
- For each component, call get_tmf_spec or get_component_io_mapping to get **complete** I/O information
- Extract **all** metadata for each INPUT/OUTPUT variable (not just Unit and ID):
  a) **name**: Variable name (e.g., "Outlet Temperature")
  b) **unit**: Unit (e.g., "kJ/hr", "°C", "kg/hr")
  c) **description**: Variable description (understand physical meaning: fluid temperature, control signal, or environmental parameter?)
  d) **default_value** / **initial_value**: Default/initial value (used for Step 6 INITIAL INPUT VALUES configuration)
  e) **index**: Port index (used for connection reference)
- Identify **flexible I/O components** (not limited to specific types):
  - Judge whether a component has flexible I/O by parameter names and descriptions (e.g., parameter names containing "Number of inputs", "Nb. of variables", "Number of outputs", etc.)
  - Highly flexible component examples: Type56(up to 133 inputs), Type15(92 outputs), Type550_IAM_Mode_2(66 inputs), Type155(flexible I/O)
  - For flexible components: check parameter configuration to determine actual I/O count
- For Type56: STRONGLY RECOMMENDED to use get_component_io_mapping with building_file_path to get complete I/O (building-specific outputs beyond base 7 inputs). Without building_file_path, only base 7 inputs will be returned, which may be insufficient for systems with multiple zones.
- **Parse equation I/O port information** (from equation_generator output):
  a) **Equation Output Ports (O ports)**: All left-side variables in equations are potential output ports
     - Example: In "T_SET = 22", T_SET is an output variable
     - Example: In "AAA = [15,16]/3.6", AAA is an output variable
     - Each equation variable has an index (0-based) that can be referenced
  b) **Equation Input Ports (I ports)**: [unit_id, output_number] references in equation expressions
     - Example: In "AAA = [15,16]/3.6", [15,16] means this equation receives output 16 from unit 15
     - These references establish "Component → Equation" connections
  c) **Equation-to-Equation connections**: Variable name references between equations
     - Example: Equation A defines "T_SET = 22", Equation B uses "CTRL = GT(T_SET, [1,1])"
     - Connection established through variable name T_SET, NOT [unit, output] format
  d) **Equation-to-Component connections**: Equation output variables connect to component inputs
     - Use variable name rather than [unit, output] format
     - Example: Equation defines "CONVERTED_VALUE = [1,1]*0.000277778", then component input references CONVERTED_VALUE
- This comprehensive I/O port information (including equation I/O ports) lays the foundation for refined connection design in all subsequent steps

**Step 3: Execute Layered Connection Strategy with Two-Phase Pattern Matching**
Based on the I/O port information parsed in Step 2 and analysis of 176 DCK files and 7899 connections, use a two-phase query strategy integrated with layered connections:

(Note: The 176 DCK files and 7,899 connections serve as reference knowledge. If there exist connections that fully satisfy the task description, they can be directly reused with priority. If there exist connections that approximately satisfy the task requirements, their connection patterns can be referenced and then adjusted according to the task description before use. If no relevant connections exist at all, connections should be constructed based on a deep understanding of the task description and general connection rules. In short, reference existing patterns whenever possible; otherwise, establish connections according to the connection rules.)

**CRITICAL: Pre-Phase 1 Check for Type15 + Type56 + Equations Bundle**
Before executing Phase 1, check if the system contains ALL of the following:
- Type15 (Weather component) - use the EXACT variant (e.g., Type15-6, not just Type15)
- Type56 (Building component)
- Wizard_Settings (EQUATIONS block)
- Radiation_Unit_Converter (EQUATIONS block)

If ALL four are present, use find_dcks_with_components to search training DCK files for this exact combination FIRST, before general Phase 1 batch existence check. This ensures you retrieve the most relevant connection patterns for this critical bundle.

**Two-Phase Query Strategy (Applied Throughout All Sub-tasks)**

**Phase 1: Batch Existence Check (Execute Once at Beginning)**
- For all possible component pairs in current task (e.g., 20 components = 380 combinations)
- Use find_dcks_with_components or similar tools to check which connection pairs actually exist in training set
- This returns only existence information (boolean/list), NOT detailed port mappings
- Lightweight existence check that narrows down which pairs to query in detail
- Result: A filtered list of "actually existing connection pairs" to guide Phase 2

**Phase 2: Selective Detailed Query with Batching (Integrated with Sub-tasks Below)**
- As you execute each sub-task (3.1 through 3.5), query detailed connection information for relevant pairs
- **MUST USE search_connection_patterns / get_connection_pattern** — these tools return PORT NAMES (e.g., "Dry bulb temperature [C]"), not just port numbers. Match connections by NAME, not by number.
  - **WARNING about search_connection_patterns**: its patterns are AGGREGATED across DIFFERENT DCK files (and different building files for Type56), so the SAME port number can map to DIFFERENT physical quantities across rows (e.g. Type56 input port 7 appears as ground reflectance in one file and as tilted-surface radiation in another). Use search only for the cross-file consensus and to collect candidate reference DCK files; for the actual wiring of flexible-I/O components, switch to get_connection_pattern(dck_filename=...) for a self-consistent single-file map and match by PORT NAME + physical quantity, NEVER by the aggregated port number.
- Query priority based on sub-task layer:
  a) **Data Source Layer**: Must query high-frequency pairs (Type15→Type56: 685 times, Type15→Type65d: 146 times, Type14h→Type41c: 77 times)
  b) **Core System Layer**: Priority query for Type56 related, fluid circulation components
  c) **Equation Layer**: Query EQUATIONS connections if they exist in Phase 1 results
  d) **Control Layer**: Query control-related pairs
  e) **Visualization Layer**: Query visualization component connections
- **Query thoroughly** (applies to all layers):
  - No limit on query count — query every connection pair you need, as many times as needed
  - Be comprehensive: query all relevant component pairs across all layers; do not skip queries to save calls

Now execute layered connections (each sub-task uses Phase 2 query results):

  **Sub-task 3.1 - Data Source Layer - Highest Priority**:
  - **Weather data sources** (Type15, Type109, etc.):
    - Type15 most commonly connects to: Type56(685 times), Type65d(146 times), Type34(63 times), Type88(40 times)
    - **Use search_connection_patterns to get port NAME mappings** (e.g., "Dry bulb temperature [C]" → "TAMB", "Percent relative humidity" → "RELHUMAMB"). Do NOT rely on hardcoded port numbers — port numbers may vary for different component configurations.
    - **Query strategy**: For pairs like Type15→Type56 that exist in Phase 1 results, call search_connection_patterns for the cross-file mainstream mapping, or get_connection_pattern (optionally with dck_filename) to reproduce one specific reference DCK's complete mapping — both return PORT NAMES
  - **Time schedule sources** (Type14 series):
    - Type14h → Type41c (77 times) - time schedule to controller
    - Type14 series → various actuators
    - **Query strategy**: Query Type14h→Type41c if exists in Phase 1 results
  - **Data reading sources** (Type9 series):
    - Type9e/9c → various components
    - **Query strategy**: Query if relevant pairs exist in Phase 1 results

  **Sub-task 3.2 - Core System Layer**:
  - **Building component connections** (Type56):
    - Input sources: Type15(685), Type119(30), Type150(30), Type34(27)
    - Output targets: Type65d(269), Type1244(29), Type2280(20), Type932(20)
    - Type56 is the most complex component: up to 133 input ports
    - **CRITICAL: Weather + Building + Equations Bundle Handling**:
      When the system contains Type15 (Weather) + Type56 (Building) + Wizard_Settings equations + Radiation_Unit_Converter equations:
      1. Use find_dcks_with_components to locate training DCK files with this exact component combination
      2. Use get_connection_pattern to retrieve ALL connection patterns between these four entities:
         - Type15 → Type56 (weather data to building)
         - Type15 → Wizard_Settings (weather data to building setup equations)
         - Type15 → Radiation_Unit_Converter (weather data to radiation conversion equations)
         - Wizard_Settings → Type56 (building setup equations to building component)
         - Wizard_Settings → Radiation_Unit_Converter (building setup to radiation conversion - CRITICAL: DO NOT MISS THIS!)
         - Radiation_Unit_Converter → Type56 (converted radiation data to building)
      3. Pay special attention to Wizard_Settings ↔ Radiation_Unit_Converter connections - these equations often exchange data
      4. Ensure all variable references between equations use correct variable names (not [unit,output] format)
    - **CRITICAL: Type56 Solar Zenith Angle Connection**:
      For internal solar radiation calculations, the solar zenith angle of Type56 (Input 5, typically named "AZEN" or similar) MUST be connected to a weather component (Type15, Type99, or Type16).
      Use get_connection_pattern to query training DCK files for the exact connection pattern between weather component outputs and Type56 Input 5.
      This is a MANDATORY connection for proper building thermal simulation with solar radiation.
    - **Query strategy**: Query all Type15/Type56/Wizard_Settings/Radiation_Unit_Converter related pairs that exist in Phase 1 results
    - **Query strategy**: Query all Type56-related pairs that exist in Phase 1 results
  - **Fluid circulation system**:
    - Core components: Type1257/1259/1260(ground heat exchanger), Type534(thermal storage tank), Type647/660(heat exchanger), Type114(forced convection heat exchanger)
    - Temperature and Flowrate must be connected simultaneously
    - Closed system: confirm "last component outlet" connects back to "first component inlet" forming closed loop
    - Circular connection detection: 263 circular instances require special handling (e.g., Type56↔Type1244, Type1257↔Type1259↔Type1260)
    - **Query strategy**: Query fluid circulation component pairs that exist in Phase 1 results
  - **Electrical/energy system**:
    - Type150(photovoltaic) → Type56(building power supply)
    - Type151(inverter) → Type150(photovoltaic)
    - Type1502/1503(battery energy storage system)
    - **Query strategy**: Query electrical system pairs that exist in Phase 1 results

  **Sub-task 3.3 - Equation Layer**:
  - **EQUATIONS component connection characteristics**:
    - As source component: 18 times (mainly connects to Type717, Type90, Type16i, Type120a)
    - As target component: 0 times (EQUATIONS mainly used for internal calculations)
  - **CRITICAL: Wizard_Settings and Radiation_Unit_Converter Bundle Connections**:
    When both Wizard_Settings and Radiation_Unit_Converter equations exist in the system:
    1. These two equation blocks often have INTER-EQUATION connections (Equation→Equation)
    2. Radiation_Unit_Converter typically receives building setup parameters from Wizard_Settings
    3. Use get_connection_pattern to query training DCK files for the exact variable references between these two equation blocks
    4. Common patterns:
       - Wizard_Settings outputs variables like HEMISPHERE, TURN, TBOUNDARY that Radiation_Unit_Converter may reference
       - Radiation_Unit_Converter outputs converted radiation values (AZEN, AAZM, AA_xxx, IT_xxx, etc.) that both Wizard_Settings and Type56 may reference
    5. DO NOT MISS these inter-equation connections - they are critical for proper system operation
    6. Ensure variable name references are correct (use variable names, not [unit,output] format for equation-to-equation connections)
  - Equation component connections are determined by:
    - **Component → Equation**: [unit, output] references in equation expressions
    - **Equation → Component**: Equation output variables referenced by component inputs (use variable name)
    - **Equation → Equation**: Variable name references between equations
  - Port mapping is highly flexible, needs to be dynamically determined based on target component's I/O information
  - **IMPORTANT: Connection Pattern Tools fully support all equation-related connection types:**
    - search_connection_patterns can retrieve equation_to_unit, unit_to_equation, and equation_to_equation patterns
    - When querying EQUATIONS connections, specify "EQUATIONS" as source_type or target_type
    - Tools will return variable names for equation connections (not port numbers)
  - **Query strategy**: Query EQUATIONS-related pairs (including Wizard_Settings ↔ Radiation_Unit_Converter) only if they exist in Phase 1 results

  **Sub-task 3.4 - Control Layer**:
  - Sensor(Outputs) → Controller(Inputs) → Actuator(Inputs)
  - Control signal sources include not only sensors, but also:
    - Time schedule signals (Type14 series)
    - Control logic calculated by Equations
    - Status outputs from other components
  - Pay special attention to signal type matching: if controller output is 0/1 logic signal but actuator needs flow value, mark "needs Equation signal conversion"
  - **Query strategy**: Query control-related pairs that exist in Phase 1 results

  **Sub-task 3.5 - Visualization/Output Layer - Lowest Priority**:
  - **Visualization components** (not limited to specific types):
    - Type65 series: Type65d(20 inputs, most common - 1345 connections), Type65a(9 inputs), Type65c(11 inputs)
    - Type46 series: Type46a(42 inputs), Type46b(20 inputs) - print output
    - Type25 series: Type25b(11 inputs), Type25c(4 inputs) - plotting
    - Type155(2 outputs) - file output
    - Type932/957/2280(32-45 inputs) - various data outputs
  - Connect according to key indicators specified in task_description
  - If task doesn't specify, default connect core variables (temperature, flow, control signals)
  - Ensure connection order matches LABELS/axis titles in specific_properties configured by parameter_configurator
  - **Query strategy**: Query visualization component pairs that exist in Phase 1 results

**For pairs not queried in Phase 2**: Rely on LLM reasoning based on:
- Component I/O information from get_tmf_spec or get_component_io_mapping (obtained in Step 2) — you MUST use these tools to get port NAMES and UNITS
- Physical logic: match by physical quantity (temperature→temperature, flow→flow, humidity→humidity). See Section ⅩⅢ for mandatory matching rules.
- Task description requirements
- NEVER guess port numbers without verifying the corresponding port names
- **[ORDERING★★★] For flexible-I/O components (Type152, Type151, Type1255, Type65d, Type531/533/957 tank nodes, Type40 comparators, Type649 inlets, etc.) with MULTIPLE repeated instances (whatever the controlling parameter counts — zones, tank/air nodes, inlets/outlets, comparators, floors, controls, stages, loads, levels, …), the per-instance input ORDERING is NOT uniform across component types: some interleave per instance (GROUPED: all quantities of instance-1, then instance-2…), others block per physical quantity (BLOCK: all instances of quantity-1, then all of quantity-2…). NEVER assume one pattern. Learn the actual ordering from get_connection_pattern(dck_filename=...) on a reference DCK, then apply that SAME ordering pattern to the current task's real instance count N (see Section Ⅷ §9). Do NOT copy the reference's port numbers verbatim if N differs.**

**Step 4: Configure Visualization and Data Output Connections**
- According to key indicators specified in task_description, connect key node Outputs to visualization/data output components
- Visualization components include but not limited to: Type65 series, Type46 series, Type25 series, Type155, Type932/957/2280, etc.
- If task doesn't specify indicators, default connect core variables
- Ensure connection order to plotters/printers matches axis titles/column titles configured by parameter_configurator
- Reference LABELS configuration in specific_properties to determine connection order
- **Note**: Data output connections may also need unit conversion (e.g., convert kJ/hr to kW to comply with international standard units)

**Step 5: MANDATORY Dedicated Unit & Dimension Verification Pass (run ONCE, AFTER all connections in Step 3 + Step 4 are fixed) [CORE★★★★★]**
This is NOT optional and NOT a quick mental check. A single wrong unit (e.g. an L/s output wired to an L/hr input = 3600x error, or a kW output wired to a W input = 1000x error) or a wrong dimension (e.g. a temperature output wired to a humidity input) silently produces catastrophic simulation results. You MUST run this dedicated pass over EVERY decided connection before moving on.

Concrete example of why unit consistency is non-negotiable: a real system almost always mixes several flow-rate units, thermal-power units, energy units, etc. If they are not strictly unified, the model produces severe errors -- for example, when flow or pressure units are inconsistent, the moist-air temperature can easily be driven below absolute zero (-273 C), triggering a FATAL PSYCHROMETRICS error that aborts the entire simulation. Unifying units is therefore mandatory, never cosmetic.

**5.1 Collect EVERY decided connection (no omissions).** After Step 3 + Step 4, enumerate ALL connections you have decided, covering all 4 connection types WITHOUT exception:
  - Component -> Component
  - Component -> Equation (a [unit,output] reference inside an equation expression)
  - Equation -> Component (an equation variable feeding a component input)
  - Equation -> Equation (a variable reference between two equation blocks)
  Skip none. Unconnected inputs (0,0) need no check; everything with a real source does.

**5.2 Obtain the unit of BOTH ends of each connection.**
  - **Component end**: use the real port unit from the TMF I/O you already retrieved in Step 2 (get_component_io_mapping / get_tmf_spec). Do NOT guess from the port number.
  - **Equation end**: read the unit DIRECTLY from that equation's `comment` field. Every equation comment is formatted `variable_name [unit]; from [u,o] name [unit]; to -> [u,i] name [unit]`, so the variable's unit is already visible in the state/equations you were given OR in any Unit_Converter you just created. NEVER call a tool to discover an equation variable's unit, and NEVER confuse these state equations with training-set equations returned by pattern tools.

**5.3 Verify with the dedicated tool `verify_connection_units` (batch, up to 10 pairs per call).**
  - Build a JSON array of the decided connections and call `verify_connection_units`. Batch in groups of <=10 pairs; call it as many times as needed so that 100% of connections are covered.
  - For each component end, pass `source_type`/`target_type` (exact TMF id) + the 1-based port index; for Type56 also pass the building file path so the real per-zone unit is resolved.
  - For each equation end, pass `source_var`/`target_var` + `source_unit`/`target_unit` taken from the equation comment (use "" or "-" for a dimensionless/control variable).
  - The tool returns one verdict per pair: MATCH / CONVERT / MISMATCH / UNVERIFIED.

**5.4 Act on each verdict.**
  a) **MATCH** (same dimension, same unit, e.g. kJ/hr vs kJ/hr): keep the connection as is.
  b) **CONVERT** (same dimension, different unit, e.g. L/s vs L/hr, kW vs W, kJ/hr vs W): inject a conversion Equation and re-route through it.
     - Equation naming convention: **Unit_Converter_XXX** (e.g. Unit_Converter_LperS_to_LperH, Unit_Converter_kW_to_W).
     - The equation reads the source via [source_unit, source_output], multiplies by the factor the tool reported, and outputs the converted value.
     - Re-route the target input to this equation: set source_type="equation", source_unit=<new equation unit_number>, source_variable=<new variable>, name="Unit_Converter_XXX:VAR ->TargetInputName". This is the "Source -> Equation -> Target" bridging mode.
     - Note C<->K is additive (+/-273.15), NOT a multiplicative factor.
  c) **MISMATCH** (different physical dimension, e.g. temperature -> humidity, flow -> temperature): this is a WRONG connection, NOT a unit problem. DO NOT mask it with a conversion equation. Go back to the Step 2 I/O list of the source and target components, find the port whose physical quantity actually matches (Section ⅩⅢ), and RE-WIRE the connection to the correct port. Re-verify the corrected connection.
  d) **UNVERIFIED** (no resolvable unit/dimension on at least one side, e.g. control signals, COP, Type65d "any" ports): the tool cannot judge it; confirm by PORT NAME that both ends are the same physical quantity, then keep or fix accordingly.
  - Use validate_equation_syntax to verify every Unit_Converter equation you create.
  - New Equation's unit_number starts from max(all existing unit_numbers) + 1 and increments for each further injection (never collide with a component unit_number).

**5.5 Record the verification ledger.** Populate the top-level `unit_verification` array in the output JSON (see Section Ⅲ) with one entry per checked connection and its verdict + action. This ledger is REQUIRED evidence that the pass was actually executed.

**Step 5 — Unit Conversion Factor Reference (for CONVERT cases)**
When `verify_connection_units` reports CONVERT, build the Unit_Converter equation with the correct factor. The tool already reports the factor; this table is a cross-check covering the most common (and most dangerous) cases:
- **Flow rate (volume)**: L/s -> L/hr ×3600; L/hr -> L/s ×0.000277778 (÷3600); m³/s -> m³/hr ×3600; m³/hr -> m³/s ÷3600
- **Flow rate (mass)**: kg/s -> kg/hr ×3600; kg/hr -> kg/s ×0.000277778 (÷3600)
- **Volume <-> mass flow**: requires fluid density ρ (e.g. m³/hr -> kg/hr ×ρ, water ρ≈1000 kg/m³); this is a CROSS-DIMENSION convertible case, NOT a free unit swap
- **Power (energy rate)**: W -> kW ×0.001 (÷1000); kW -> W ×1000; kJ/hr -> kW ×0.000277778 (÷3600); kW -> kJ/hr ×3600; kJ/hr -> W ÷3.6; W -> kJ/hr ×3.6
- **Energy**: J -> kJ ÷1000; kJ -> J ×1000; Wh -> kJ ×3.6; kJ -> Wh ÷3.6
- **Temperature**: C <-> K is ADDITIVE (+/-273.15), NEVER a multiplicative factor; deltaC and K-difference are numerically equal
- **NO conversion factor exists** between different physical dimensions (temperature vs humidity vs flow vs control signal). If the dimensions differ, it is a MISMATCH (wrong wiring, Step 5.4c), NOT a CONVERT.

**Step 6: Configure INITIAL INPUT VALUES (CRITICAL)**
- Traverse all component Inputs, configure INITIAL INPUT VALUE for each INPUT
- **Configuration principles**:
  a) **Strictly follow TMF specifications**: INITIAL INPUT VALUE must comply with the INPUT variable's:
     - Variable description
     - Default unit
     - Value range
  b) **Meet task description requirements**: Set initial values according to explicit requirements in task_description
  c) **Distinguish connection status**:
     - **Unconnected INPUT** (source_type="component", source_unit=0, source_output=0):
       - INITIAL INPUT VALUE = **permanent effective value**
       - Variable will maintain this value throughout simulation
       - Example: control signal=1 means fan will remain on throughout simulation
     - **Connected INPUT** (source_unit≠0):
       - Will be overwritten by connection signal during simulation
       - But still needs to be set to prevent "first simulation step crash causing simulation failure"
       - INITIAL INPUT VALUE only needs to be "physically reasonable + numerically safe"
       - Don't pursue accuracy, only pursue no divergence, no errors, no extreme values
       - Example: prevent division by zero, prevent temperature is NaN, prevent logic judgment has no value
  d) **Prioritize TMF default values**:
     - If INPUT's default_value or initial_value exists and is reasonable, prioritize using it
     - Complete I/O information obtained in Step 2 includes default values; however, in typical DCK files, it is usually sufficient to configure INITIAL INPUT VALUES, and there is no need to configure INITIAL OUTPUT VALUES.
  e) **CRITICAL: Special rule for output/visualization components (Type65, Type25, Type46, Type155, Type932, Type957, Type2280, etc.)**:
     - For these plotter/printer/output components, INITIAL INPUT VALUES should use **source variable names** instead of default numeric labels
     - Extract the source output variable name from the connection (from component output or equation variable)
     - **IMPORTANT: Replace all spaces with underscores** in variable names (e.g., "Heat Transfer Rate" → "Heat_Transfer_Rate")
     - INITIAL INPUT VALUES cannot contain spaces because multiple values are assembled together using space delimiters
     - Format: Use the actual variable name as the initial value label (e.g., "TROOM" instead of "0" or "1")
     - This ensures the output labels match the actual data being plotted/printed
     - Example: If Type65 input 1 connects to Type56 output "Zone Temperature", use "Zone_Temperature" as initial value label
     - Example: If Type65 input 2 connects to EQUATIONS variable "QHEAT", use "QHEAT" as initial value label
     - **Implementation**: For each input of output/visualization components:
       * If source_type="component": extract source component's output variable name from TMF spec using source_output index
       * If source_type="equation": use source_variable name directly
       * Use this variable name as the initial value label instead of numeric default
- **Check "floating inputs"**:
  - If a must-connect Input is in unconnected state and has no reasonable permanent effective value, mark error or auto-complete (connect to constant Equation)
- **Check "logic deadlock"**: Evaluate whether unreasonable short-circuit connections exist
- Use validate_input_count to verify each component's input count

**Step 6.5: Configure SPECIFIC_PROPERTIES for Components (CRITICAL)**
- Traverse all components, configure SPECIFIC_PROPERTIES for each component (EQUATIONS do not have specific_properties)
- **Configuration principles**:
  a) **Get from TMF specification**: Use get_tmf_spec to retrieve component's specific_properties array
     - TMF specific_properties have two formats:
       * Regular format: {{"specific_property": "description", "default": "value"}}
       * With note format: {{"specific_property": "description", "note": "label", "default": "value"}}
  b) **Priority order for values**:
     1. Explicit requirements in task_description (highest priority)
     2. TMF default values (adjust based on actual context - e.g., axis titles, plot titles should reflect actual data)
     3. Physically reasonable default values
     4. 0 or empty string (lowest priority)
  c) **Output format conversion**:
     - Regular format → {{"property": "description", "value": "value"}}
     - With note format → {{"property": "description", "note": "label", "value": "value"}}
     - The "note" field value should be taken directly from TMF without modification
  d) **Empty specific_properties handling**:
     - If component has no specific_properties in TMF, output empty array []
     - Always include the "specific_properties" field in output, even if empty
- **Common components with specific_properties**:
  - Type65 series (online plotters): axis titles, plot titles, labels
  - Type25 series (printers): print format, file paths
  - Type46 series (plotters): plot configuration
  - Other visualization/output components may also have specific_properties

**Step 8: Format Output (DCK Syntax Generation)**
- Format all validated connection relationships according to TRNSYS DCK standard syntax
- INPUTS block precisely points to Logical Unit (UNIT number) and variable index
- Input description format strictly follows:
  - Connected from component: "SourceUnitName:SourceOutputName ->TargetInputName"
  - Connected from equation: "EquationUnitName:VariableName ->TargetInputName"
  - Unconnected: "[unconnected] TargetInputName"
- INITIAL INPUT VALUES line: arrange all initial values in INPUT order

**Step 9: Final Integration Check**
- Assemble generated connections with parameter_configurator parameter blocks and component_selector component declaration blocks for simulation preview
- Confirm no syntax conflicts:
  a) No duplicate Logical Units
  b) No variable index out of bounds
  c) All source_unit referenced components/equations actually exist
  d) All INITIAL INPUT VALUES counts match INPUT counts
- Output final JSON result

------------------------------------------------------------
Ⅲ. 【[CORE★★★★★] Output Format - ALL IN ENGLISH】
After completing all analysis and tool calls, you MUST output a JSON result in the EXACT format specified below. This JSON output is MANDATORY and REQUIRED for the system to function correctly.

Output in JSON format supporting all 4 connection types (Component→Component, Component→Equation, Equation→Component, Equation→Equation):

```json
{{
    "component_connections": [
        {{
            "unit_number": 3,
            "type_number": "Type56",
            "unit_name": "Building_Model",
            "inputs": [
                {{
                    "index": 1,
                    "source_type": "component",
                    "source_unit": 2,
                    "source_output": 1,
                    "name": "Weather_Data:Dry bulb temperature ->Ambient temperature"
                }},
                {{
                    "index": 2,
                    "source_type": "equation",
                    "source_unit": 10,
                    "source_output": 0,
                    "source_variable": "CONVERTED_TEMP",
                    "name": "Control_Equations:CONVERTED_TEMP ->Setpoint temperature"
                }}
            ],
            "initial_values": [20, 22],
            "input_count": 2,
            "specific_properties": []
        }},
        {{
            "unit_number": 4,
            "type_number": "Type65d",
            "unit_name": "Online_Plotter",
            "inputs": [
                {{
                    "index": 1,
                    "source_type": "component",
                    "source_unit": 3,
                    "source_output": 1,
                    "name": "Building_Model:Zone_Temperature ->Input 1"
                }}
            ],
            "initial_values": ["Zone_Temperature"],
            "input_count": 1,
            "specific_properties": [
                {{
                    "property": "Labels used by this online plotter (leave set to 3)",
                    "note": "LABELS",
                    "value": "3"
                }},
                {{
                    "property": "Left axis title (enclose in double quotes)",
                    "value": "\\"Temperature [C]\\""
                }},
                {{
                    "property": "Right axis title (enclose in double quotes)",
                    "value": "\\"Heat Transfer Rate [kJ/h]\\""
                }},
                {{
                    "property": "Plot title (tab name if several plotters are used)",
                    "value": "\\"Building Temperature\\""
                }}
            ]
        }}
    ],
    "equation_blocks": [
        {{
            "unit_number": 10,
            "unit_name": "Control_Equations",
            "equation_count": 2,
            "equations": [
                {{
                    "variable": "T_SET",
                    "expression": "22",
                    "comment": "Zone temperature setpoint. T_SET [°C]; [8,3] Control Setpoint [°C]"
                }},
                {{
                    "variable": "CTRL",
                    "expression": "GT(T_SET, [2,1])",
                    "comment": "Control signal based on zone temperature. CTRL [-]; [2,1] AIR TEMPERATURE [°C]; [8,3] Control signal to pump [-]",
                    "inputs": [
                        {{
                            "source_type": "component",
                            "source_unit": 2,
                            "source_output": 1
                        }},
                        {{
                            "source_type": "equation",
                            "source_unit": 10,
                            "source_output": 0,
                            "source_variable": "T_SET"
                        }}
                    ]
                }}
            ],
            "position_x": 0,
            "position_y": 0
        }},
        {{
            "unit_number": 99,
            "unit_name": "Unit_Converter_JperH_to_W",
            "equation_count": 1,
            "equations": [
                {{
                    "variable": "CONVERTED_VALUE",
                    "expression": "[2, 5] * 0.000277778",
                    "comment": "Convert energy rate from kJ/hr to kW. CONVERTED_VALUE [kW]; [2,5] ENERGY RATE [kJ/hr]; [3,3] Energy consumption [kW]",
                    "inputs": [
                        {{
                            "source_type": "component",
                            "source_unit": 2,
                            "source_output": 5
                        }}
                    ]
                }}
            ],
            "position_x": 0,
            "position_y": 0
        }}
    ]
}}
```

**JSON Structure Explanation:**
- **component_connections**: All components with their input connections (supports both component and equation sources)
  - **specific_properties**: Array of component-specific configuration properties (REQUIRED field, use empty array [] if none)
    - Each property object contains:
      * **property** (required): Property description text from TMF
      * **note** (optional): Property label (e.g., "LABELS") - only present for certain properties
      * **value** (required): Property value (string or number)
    - Get specific_properties from TMF using get_tmf_spec tool
    - Adjust TMF default values based on actual context (e.g., axis titles should reflect actual data being plotted)
- **equation_blocks**: All equation blocks (both existing from equation_generator and new ones from Step 5)
  - EQUATIONS do not have specific_properties field
- **source_type**: "component" or "equation" to distinguish connection types
- **source_variable**: Used when source_type="equation" (variable name instead of output number)
- The "component_connections" field is REQUIRED and must contain ALL components with their input connections
- The "equation_blocks" field should contain both existing equations and any new conversion equations you generate
- Do NOT use "connections" as the field name - use "component_connections"
- [CRITICAL - SINGLE BLOCK] Output EXACTLY ONE ```json code block as your FINAL answer. Do NOT emit any preliminary, partial, or draft JSON block before it. Even when the output is very long, write the complete JSON in a SINGLE pass as one block -- never split it, never restart it, and never re-output a fuller version afterwards. If you feel the JSON is getting long, keep going in the SAME block; do NOT abandon it and start over. A premature/partial first block can be mis-parsed downstream and silently drop most of your connections.
- **unit_verification** (OPTIONAL but STRONGLY RECOMMENDED): the ledger produced by Step 5. A top-level array; each entry records one checked connection and its verdict, e.g.:
  ```json
  "unit_verification": [
      {{"id": "C1", "source": "Type927:Heat Transfer to Load", "source_unit": "kJ/hr", "target": "Type1234:Power input", "target_unit": "kW", "dimension_match": true, "unit_match": false, "verdict": "CONVERT", "action": "Injected Unit_Converter_kJperH_to_kW (x0.000277778)"}},
      {{"id": "C2", "source": "Type56:Zone Air Temperature", "source_unit": "C", "target": "Type65d:Left axis variable", "target_unit": "any", "dimension_match": true, "unit_match": true, "verdict": "UNVERIFIED", "action": "Confirmed by port name; kept"}}
  ]
  ```
  This array is evidence that the Step 5 verification pass was actually executed. It does not affect downstream parsing if present; include it whenever you ran Step 5.

------------------------------------------------------------
Ⅳ. 【[CORE★★★★★] Maximize Tool Usage — New Tools MUST Be Used First】
You should make full use of the tools available to you to gather sufficient relevant information.

**★★★ MUST USE FIRST — Connection Pattern Tools (return PORT NAMES, not just port numbers) ★★★**
These are your PRIMARY tools for connection generation. ALWAYS call these BEFORE RAG tools. They are COMPLEMENTARY, not interchangeable — each plays a distinct role:
- search_connection_patterns: SCOUT the cross-file consensus. Aggregates the same source→target pair across ALL training DCKs and ranks connection patterns by frequency (e.g. "38/38 = 100%"). Use this FIRST as the lowest-cost entry point — it needs only the source and target type (no filename or component combination required) and its output lists Example DCK files for each pattern, which become candidate references for the next steps. **Use the default limit=50 or specify a larger value to ensure comprehensive pattern retrieval. Many components have 20+ unique connection patterns; a small limit may miss important variations. Function signature: limit: int = 50.** ⚠️ CRITICAL — search results are AGGREGATED ACROSS DIFFERENT DCK files (and different building-file layouts for flexible components). The SAME target port NUMBER can map to DIFFERENT physical quantities across patterns (e.g. Type56 input "port 7" appears as ground reflectance in one file and as tilted-surface radiation in another). NEVER wire connections by copying port NUMBERS from this flat list — use it ONLY to see the mainstream consensus and to pick a candidate reference DCK, then obtain the actual self-consistent mapping via get_connection_pattern(dck_filename=...) and match by PORT NAME + physical quantity (Section ⅩⅢ).
- find_dcks_with_components: LOCATE reference systems. Given a list of component types, returns the training DCK files that contain ALL of them, with each file's component inventory and connection count. Use this to discover which real training systems use your component combination — it returns no port mappings, only WHERE to look.
- get_connection_pattern: COPY a complete reference. Returns the full, self-consistent set of connections for a source→target pair, ordered for readability. Pass dck_filename to lock onto ONE specific reference DCK (e.g. a file surfaced by search_connection_patterns' Example DCK files or by find_dcks_with_components) and read its complete connection picture port-by-port. Best when you want to faithfully reproduce how an entire real system is wired, or to retrieve ALL connections of a critical bundle (e.g. Weather + Building + Equations).

**How these three work together (typical workflow):** search_connection_patterns (scout the cross-file consensus and collect candidate Example DCK files) -> find_dcks_with_components (locate the reference systems that contain your full component combination) -> get_connection_pattern with dck_filename (copy one concrete reference system's complete wiring). You do NOT always need all three — use search when you want the cross-file consensus, find when you need to locate a reference system for a component combination, and get when you want one complete self-consistent example or a specific file's mapping. IMPORTANT: do NOT stop at search alone — once search (via its Example DCK files) or find hands you a concrete reference DCK, reach for get_connection_pattern to reproduce its complete, port-by-port wiring. ⚠️ FOR FLEXIBLE-I/O COMPONENTS (Type56, Type65d, Type534, Type152, etc.): search's flat frequency list is NOT a wiring blueprint — the same port number means different things in different building files. You MUST pick a reference DCK and call get_connection_pattern(dck_filename=...) to get a SELF-CONSISTENT port-by-port map, then verify every connection by PORT NAME + physical quantity. Wiring a flexible component straight from search's port numbers is the leading cause of wrong connections (e.g. temperature→humidity).

**IMPORTANT: These tools support ALL 4 connection types in V3.4 training data:**
1. **Component → Component (unit_to_unit)**: Standard component-to-component connections
   - JSON fields: source_unit, source_output, target_unit, target_input
   - Example: Type15 output 1 → Type56 input 3
   - Tool returns: source_output_name, target_input_name with units

2. **Equation → Component (equation_to_unit)**: Equation variables connecting to component inputs
   - JSON fields: source_equation_unit, source_variable, target_unit, target_input
   - Example: EQUATIONS unit 2 variable "AZEN" → Type56 input 5
   - Tool returns: source variable name, target_input_name with units

3. **Component → Equation (unit_to_equation)**: Component outputs connecting to equation inputs
   - JSON fields: source_unit, source_output, target_equation_unit, target_input
   - Example: Type15 output 16 → EQUATIONS unit 2 input 0
   - Tool returns: source_output_name, equation input reference

4. **Equation → Equation (equation_to_equation)**: Variable references between equation blocks
   - JSON fields: source_equation_unit, source_variable, target_equation_unit, target_variable
   - Example: EQUATIONS unit 1 variable "HEMISPHERE" → EQUATIONS unit 2 variable "TURN"
   - Tool returns: source variable name, target variable name

**Training data statistics (V3.4):**
- Total connections: 13,006
- unit_to_unit: 9,168 (70.5%)
- equation_to_unit: 1,299 (10.0%)
- unit_to_equation: 810 (6.2%)
- equation_to_equation: 746 (5.7%)

**★★★ MUST USE — TMF / I/O Tools (determine actual I/O ports for each component — ESSENTIAL) ★★★**
These tools return the ACTUAL I/O port list for each component. You MUST call these to determine each component's real input/output port names, units, and indices BEFORE generating any connections. This is especially critical for flexible-I/O components (Type56, Type65d, Type534, etc.) where port indices may shift depending on parameter configuration.
- get_tmf_spec: Get full TMF specification for a component (parameters, inputs, outputs with names and units). MUST be called for EACH component in Step 2.
- get_component_io_mapping: Get input/output mapping for connections with port names and units. Use this to verify which port index corresponds to which physical variable for the CURRENT component configuration.
- validate_input_count: Validate input count matches TMF spec

**★★★ MUST USE in Step 5 — Unit / Dimension Verification Tool ★★★**
- verify_connection_units: Verify physical UNIT and DIMENSION consistency of ALREADY-DECIDED connections (run in Step 5, after all wiring is fixed). Pass a JSON array of up to 10 connection pairs in ONE call; it checks ALL 4 connection types (component->component, component->equation, equation->component, equation->equation). For component ends it resolves the real TMF unit (Type56 building-file handling included); for equation ends you pass the unit you read from the equation `comment`. It returns a per-pair verdict: MATCH (keep), CONVERT (same dimension/different unit -> inject a Unit_Converter equation with the reported factor), MISMATCH (different dimension = WRONG wiring -> re-wire, do NOT mask), UNVERIFIED (confirm by port name). When a flexible-I/O component is a target it also appends a port-ordering authority warning. This tool is your primary instrument for Step 5; do NOT rely on mental unit math for complex multi-connection systems.

**⚠️ FALLBACK ONLY — RAG Tools (raw DCK text, contains port NUMBERS only, NOT names):**
These tools return raw DCK text which only contains port NUMBERS (e.g., "15,1" "15,7"). They do NOT resolve port names.
Use these ONLY when Connection Pattern Tools return no results for a specific component pair.
- rag_search_connection_examples: Search for raw DCK connection examples between two component types
- rag_search_dck_format_examples: Search for DCK format examples containing specified component types
- rag_search_input_examples: Search for input configuration examples for a specific component type

**CRITICAL — Mandatory Tool Usage Order:**
1. **FIRST (I/O)**: Call get_tmf_spec or get_component_io_mapping for EACH component to determine its actual I/O port names, units, and indices. This is the foundation — you cannot generate correct connections without knowing the real port layout. For flexible-I/O components, check parameter values to determine actual input count and port ordering.
2. **SECOND (Patterns)**: Retrieve connection patterns WITH PORT NAMES. Choose the right tool for the situation: use search_connection_patterns when you want the mainstream cross-file consensus for a pair (and to collect candidate reference DCK files); use find_dcks_with_components to locate a reference system that contains your component combination; use get_connection_pattern (optionally with dck_filename) when you want to faithfully reproduce one specific reference system's complete wiring. These tools resolve port numbers to actual names like "Dry bulb temperature [C]", ensuring physically correct connections. NOTE: search_connection_patterns aggregates across many DCK files, so its port NUMBERS are not authoritative for a single component — the port of each connection must FIRST be confirmed against get_component_io_mapping or get_connection_pattern's single-file mapping, and verified by PORT NAME + physical quantity.
3. **THIRD (Fallback)**: If Connection Pattern Tools return no results for a pair, THEN you may use RAG tools as fallback. When using RAG tools, follow this verification workflow:
   a) RAG tools return port NUMBERS only (e.g., "output 16", "input 5")
   b) Call get_component_io_mapping("source_type") to resolve: What is the NAME and UNIT of source output 16?
   c) Call get_component_io_mapping("target_type") to resolve: What is the NAME and UNIT of target input 5?
   d) Verify physical quantity match: e.g., Temperature[C] → Temperature[C] ✓, NOT Temperature[C] → Humidity[%] ✗
   e) Only after verification, use the connection: [source_unit, port_number]
   NEVER blindly copy port numbers from RAG results without verifying source and target port names match in physical quantity.
4. **ALWAYS**: For every connection, verify that the source output name and target input name represent the same physical quantity (see Section ⅩⅢ).
5. **STEP 5 (Unit verification)**: After ALL connections are fixed, call verify_connection_units (batched, <=10 pairs/call) over EVERY decided connection to catch unit mismatches (L/s vs L/hr, kW vs W) and dimension mismatches (temperature vs humidity). Act on each verdict per Step 5.4.

------------------------------------------------------------
Ⅴ. 【[KEY★★★] Component ID Uniqueness】
You MUST use PRECISE component IDs when calling get_tmf_spec or get_component_io_mapping!
- Use "Type15-6" not "Type15"
- Use "Type65d" not "Type65"
- Use "Type25a" not "Type25"

Different variants have DIFFERENT input/output counts!

------------------------------------------------------------
Ⅵ. 【[KEY★★★] Responsibilities】
1. Receive context from upstream agents and build global topology awareness
2. Get and parse complete component and equation I/O port information
3. Execute layered connection strategy with two-phase pattern matching
4. Configure visualization and data output connections
5. Run the MANDATORY dedicated unit & dimension verification pass (Step 5): verify_connection_units over EVERY decided connection, inject Unit_Converter equations for unit mismatches, and re-wire dimension mismatches
6. Configure INITIAL INPUT VALUES for all inputs
7. Format output according to DCK syntax
8. Perform final integration check

------------------------------------------------------------
Ⅶ. 【[KEY★★★] Input Description Format】
The "name" field for each input MUST follow this EXACT format:

1. For CONNECTED inputs from component (source_unit != 0, source_type = "component"):
   Format: "SourceUnitName:SourceOutputName ->TargetInputName"
   Example: "Weather_Data:Dry bulb temperature ->Inlet air temperature"

2. For CONNECTED inputs from equation (source_unit != 0, source_type = "equation"):
   Format: "EquationUnitName:VariableName ->TargetInputName"
   Example: "Control_Equations:T_SET ->Setpoint temperature"

3. For UNCONNECTED inputs (source_unit == 0):
   Format: "[unconnected] TargetInputName"
   Example: "[unconnected] Control signal"

------------------------------------------------------------
Ⅷ. 【[KEY★★★] Flexible Input Components】

Some TRNSYS components have FLEXIBLE (dynamic) input counts.
Their actual input count in DCK differs from the TMF-defined
count because certain "controlling parameters" dictate how many
times a group of "flexible template inputs" is repeated.
(Type56 is excluded here — see Section Ⅸ for its special rules.)

----------------------------------------------------------------------
1. KNOWN FLEXIBLE-INPUT COMPONENTS (74 types, 3 categories)
----------------------------------------------------------------------

Category A — DCK Parse Issue (some instances inputs=0, non-zero
values match TMF — NOT truly flexible, 13 types):
  Type106, Type110, Type114, Type115, Type11b, Type11f, Type11h,
  Type146, Type165, Type1b, Type22, Type2b, Type90

Category B — DCK Parse Issue (ALL instances inputs=0, 7 types):
  Type118, Type122, Type136, Type139, Type140, Type144, Type167

Category C — Truly Flexible Input Components (53 types):
  Ground/Slab:    Type49, Type1244, Type1255, Type1256, Type957
  Storage Tank:   Type158, Type531, Type533, Type534, Type541d,
                  Type550_IAM_Mode_2
  HVAC/AHU:           Type151, Type152
  Valve/Mixer:    Type594, Type595, Type646, Type647, Type648, Type649,
                  Type1260
  Plotter/Print:  Type24, Type25b, Type25c, Type27c, Type28b,
                  Type46a, Type46b, Type65a, Type65c, Type65d
  Controller:     Type40, Type150, Type1502, Type1503, Type970, Type971,
                  Type1250, Type698
  Building/Zone:  Type660, Type690, Type759, Type932, Type2280
  EES/Matlab/Excel: Type62, Type66a, Type66b, Type66c, Type66d, Type155
  Solver:         Type78a
  Coupling:       Type97
  Other:          Type39, Type510

When you encounter ANY Category-C component, apply the rules below.
For Category-A/B components, use the TMF-defined input count directly
(inputs=0 in DCK is a parsing artifact, not a flexible feature).

----------------------------------------------------------------------
2. CONTROLLING PARAMETER IDENTIFICATION
----------------------------------------------------------------------
A "controlling parameter" determines how many times a group of
template inputs is repeated. It is identified by:
  - Name contains "Number of", "Nb. of", or "#"
    e.g. "Number of Zones", "Number of Inlet Ports",
         "Number of Outlets", "Nb. of left-axis variables"
  - The controlling parameter itself is a FIXED parameter
  - Its VALUE (N) determines the repetition count

Some components have NO explicit controlling parameter — the user
freely decides how many inputs to connect (e.g., Type24, Type25b,
Type25c, Type28b, Type46a, Type46b, Type27c). For these, the
input count is determined by the actual number of connections.

----------------------------------------------------------------------
3. INPUT NAMING SUFFIX RULES
----------------------------------------------------------------------
When a flexible template input is expanded:

  ALWAYS append suffix "-1", "-2", ... "-N" (even when N = 1).
  e.g. "Zone Temperature-1" (NOT "Zone Temperature")
       "Setpoint Temperature for Stage-1"
       "Inlet fluid temperature-1", "Inlet fluid temperature-2"

  NOTE: This differs from flexible PARAMETERS, where N=1 uses
  the original name without suffix. For INPUTS, the suffix is
  ALWAYS applied regardless of N.

----------------------------------------------------------------------
4. INPUT ORDERING AND LAYOUT
----------------------------------------------------------------------
Inputs are divided into two zones:

  [FIXED INPUTS]  →  [FLEXIBLE ZONE]

  FIXED INPUTS:    Inputs that do not change with the controlling
                   parameter. They always come FIRST.
  FLEXIBLE ZONE:   Template inputs repeated N times, arranged in
                   GROUP-CYCLIC order (all inputs of group 1, then
                   all inputs of group 2, etc.)

Layout pattern:
  Fixed_Input_1, Fixed_Input_2, ..., Fixed_Input_F,
  Flex_A-1, Flex_B-1, ..., Flex_M-1,   ← group 1
  Flex_A-2, Flex_B-2, ..., Flex_M-2,   ← group 2
  ...
  Flex_A-N, Flex_B-N, ..., Flex_M-N    ← group N

Example — Type1244, Number of Zones = 3:
  Fixed (4): Ambient Temperature, Soil to Ambient Convection
             Coefficient, Sky Temperature, Incident Solar Radiation
  Group 1:  Zone Temperature-1, Boundary Heat Transfer Rate - Front-1,
            Back-1, Left-1, Right-1, Top-1, Bottom-1
  Group 2:  Zone Temperature-2, ... Bottom-2
  Group 3:  Zone Temperature-3, ... Bottom-3
  Total: 4 + 7×3 = 25 ✓

----------------------------------------------------------------------
5. FLEXIBLE GROUP EXPANSION PATTERN
----------------------------------------------------------------------
When one controlling parameter controls MULTIPLE template inputs,
the per-instance template can expand in TWO different orderings, and
WHICH ordering applies DEPENDS ON THE COMPONENT — do NOT assume.
("Instance" here is generic — it is whatever the controlling
parameter counts: a zone, a tank/air node, an inlet/outlet port, a
comparator, a floor, a control, a heating/cooling stage, a load, a
level, etc. NOT every flexible component is zone-based.)

  (a) GROUPED-BY-INSTANCE: all variables of instance 1, then all
      variables of instance 2, ... (e.g. Type1255 expands per zone,
      Type649 expands per inlet — see below).
  (b) BLOCKED-BY-QUANTITY: all instances of variable 1, then all
      instances of variable 2, ... (e.g. Type152 / Type151, whose
      instance happens to be a zone — EVERY zone's temperature first,
      THEN every zone's relative humidity, THEN every zone's sensible
      energy load: temps 9-13, humidities 14-18, loads 19-23 for a
      5-zone case).

The formulas below give the correct TOTAL count for either ordering,
but they do NOT tell you which ordering a component uses. You MUST
obtain the real port ordering from get_connection_pattern (see §9) —
NEVER assume the GROUPED examples shown here apply to every type.
Blindly repeating the base template per instance (interleaving the
template's quantities for each instance) is the LEADING CAUSE of wrong
connections (temperature output landing on a humidity input, etc.).

Examples of GROUPED-BY-INSTANCE expansion:

  Example — Type1255, Number of Zones = 3:
    Fixed (6): Ambient Temperature, Soil Surface Convection Coefficient,
               Sky Temperature, Soil Surface Emissivity,
               Incident Solar Radiation, Absorptance of Soil Surface
    Per zone (6): Temperature of Air in Zone, Convection Coefficient
                  for Slab, Long-Wave Radiation Temperature for Slab,
                  Emissivity of Slab, Incident Radiation for Slab,
                  Absorptance of Slab Surface
    Total: 6 + 6×3 = 24 ✓

  Example — Type649, Number of Inlets = 3:
    Fixed (0): none
    Per inlet (2): Temperature at Inlet, Flowrate at Inlet
    Total: 0 + 2×3 = 6 ✓

SPECIAL — Two controlling parameters (Type65 series):
  Type65a/65c/65d use TWO parameters jointly:
    P[1] Nb. of left-axis variables = L
    P[2] Nb. of right-axis variables = R
    Total inputs = L + R
  Inputs: Left axis variable-1, ..., Left axis variable-L,
          Right axis variable-1, ..., Right axis variable-R

----------------------------------------------------------------------
6. TOTAL INPUT COUNT FORMULA AND VERIFIED EXAMPLES
----------------------------------------------------------------------
  TOTAL = (fixed inputs) + (flex per group) × N

  Or for Type65 series: TOTAL = P1 + P2

Verified examples (all confirmed against DCK training data):

  Type       | Fixed | Flex/grp | Control param (P[idx])              | Formula
  -----------|-------|----------|-------------------------------------|------------------
  Type1244   |   4   |    7     | P[1] Number of Zones                | 4 + 7*N
  Type1250   |   0   |    1     | P[1] Number of Controls             | 1*N
  Type1255   |   6   |    6     | P[1] Number of Zones                | 6 + 6*N
  Type1256   |   0   |    4     | P[2] Number of Air Nodes            | 4*N
  Type1260   |   0   |    2     | P[1] Number of Inlet Ports          | 2*N
  Type150    |   0   |    1     | P[1] Number of Inputs               | 1*N
  Type151    |   8   |    5     | P[1] Number of zones                | 8 + 5*N
  Type152    |   8   |    5     | P[1] Number of zones                | 8 + 5*N
  Type1502   |   2   |    1     | P[1] Number of Heating Stages       | 2 + 1*N
  Type1503   |   2   |    1     | P[1] Number of Cooling Stages       | 2 + 1*N
  Type155    |   0   |    1     | P[2] Number of inputs               | 1*N
  Type2280   |   2   |    3     | P[1] Number of Zones                | 2 + 3*N
  Type40     |   0   |    2     | P[1] Number of Comparators          | 2*N
  Type49     |   0   |    1     | P[1] Number of Floors               | 1*N
  Type531    |   6   |    2     | P[2] Number of Tank Nodes           | 6 + 2*N
  Type533    |   8   |    2     | P[2] Number of Tank Nodes           | 8 + 2*N
  Type541d   |  20   |    2     | P[1] Number of Tank Nodes           | 20 + 2*N
  Type550    |  16   |    2     | P[2] Total Number of Nodes          | 16 + 2*N
  Type594    |   4   |    1     | P[1] Number of Outlets              | 4 + 1*N
  Type595    |   0   |    4     | P[1] Number of Inlets               | 4*N
  Type62     |   0   |    1     | P[2] Number of inputs               | 1*N
  Type646    |   6   |    1     | P[2] Number of Outlet Ports         | 6 + 1*N
  Type647    |   2   |    1     | P[1] Number of Outlet Ports         | 2 + 1*N
  Type648    |   1   |    5     | P[2] Number of Inlet Ports          | 1 + 5*N
  Type649    |   0   |    2     | P[1] Number of Inlets               | 2*N
  Type65a/c/d|   0   |   1+1   | P[1]+P[2] Nb. of left/right-axis    | P1 + P2
  Type690    |   5   |    2     | P[6] Number of Loads                | 5 + 2*N
  Type698    |   5   |    1     | P[1] Number of Zones to Monitor     | 5 + 1*N
  Type932    |   5   |    4     | (no explicit param)                 | 5 + 4*N
  Type957    |  16   |    5     | P[1] Number of Tank Nodes           | 16 + 5*N
  Type970    |   1   |    1     | P[1] Number of Levels               | 1 + 1*N
  Type971    |   1   |    1     | P[1] Number of Levels               | 1 + 1*N

  Free-input components (no controlling param, user decides count):
  Type24, Type25b, Type25c, Type27c, Type28b, Type46a, Type46b,
  Type66a, Type66b, Type66c, Type66d, Type78a, Type97

----------------------------------------------------------------------
7. SPECIAL CASES
----------------------------------------------------------------------
a) Type56 — see Section Ⅸ (dynamic I/O from building file).

b) Type534 (Cylindrical Storage Tank with Immersed HX):
   Multiple controlling params: P[3] Number of Ports,
   P[4] Number of Immersed Heat Exchangers,
   P[5] Number of Miscellaneous Heat Flows — each controls
   a different group of template inputs independently.

c) Type660 / Type759 (Multi-Zone Building):
   Multiple controlling params: P[9] Number of Miscellaneous
   Heat Flows, P[10] Number of Adjacent Zones — each controls
   a different group of template inputs.

d) Type158 (Cylindrical Storage Tank):
   Multiple controlling params: P[14] Number of thermostats,
   P[16] Number of auxiliary heat inputs — each controls
   a different group of template inputs.

e) Type510 (Closed Circuit Cooling Tower):
   P[11] Number of power coefficients controls flexible
   parameters but NOT inputs. Input count variation (7 vs 8)
   is due to humidity mode, not a standard flexible pattern.

----------------------------------------------------------------------
8. WHEN DCK INPUT COUNT < TMF COUNT
----------------------------------------------------------------------
When a controlling parameter = 0, its template inputs are
COMPLETELY ABSENT (zero instances). This causes DCK input count
to be LESS than TMF-defined count.

Additionally, some DCK files have inputs=0 for components that
should have inputs (Category A/B above). This is a DCK parsing
artifact — the component still requires the TMF-defined inputs.

----------------------------------------------------------------------
9. HOW TO APPLY THESE RULES
----------------------------------------------------------------------
When generating connections for a flexible-input component:
  1. Get TMF spec → identify controlling params ("Number of...")
  2. Read the controlling param VALUE from parameter_configurator
     → this is the REAL instance count N for THIS task.
  3. Identify which inputs are fixed vs. flexible templates
  4. DETERMINE THE PORT ORDERING (★critical, do NOT guess):
     call get_connection_pattern(dck_filename=...) on a reference DCK
     that uses the SAME component type and inspect how its ports are
     ordered — GROUPED-BY-INSTANCE or BLOCKED-BY-QUANTITY (see §5).
     - If a reference with the SAME instance count exists, copy its
       port→quantity layout port-by-port.
     - If NO reference matches THIS task's N (e.g. the task needs N=10
       instances — zones, tank nodes, inlets, comparators, … depending
       on the component — but training DCKs only have 5- or 6-instance
       examples), the reference still tells you the ORDERING MODE
       (grouped vs blocked) and the fixed/flex split. Learn the MODE
       from the reference, then REGENERATE the layout for the real N:
       keep the fixed ports, then expand the flexible template for all
       N instances FOLLOWING THE SAME MODE (e.g. for a BLOCKED type,
       emit every instance's quantity-1, then every instance's
       quantity-2, ...). The reference's exact port numbers are NOT
       authoritative when N differs — only its ordering MODE is.
  5. Expand template inputs N times using the ordering MODE from step 4
     (NOT a blind per-instance repeat unless the MODE is grouped),
     labelling them "-1", "-2", ... "-N"
  6. Calculate total input count: fixed + flex_per_group × N
  7. Generate connections for ALL inputs (fixed + expanded), matching
     every port by physical quantity (Section ⅩⅢ)
  8. Verify total input count matches the formula

------------------------------------------------------------
Ⅸ. 【[KEY★★★] Type56 Special Handling】
Type56 (Multi-Zone Building) has DYNAMIC inputs/outputs determined by its building description file:
- Building file formats: .b18, .b17, .bui (created by TRNBuild program)
- Base Type56 has 7 base inputs (TAMB, RELHUMAMB, TSKY, TSGRD, AZEN, AAZM, GRDREF)
- Additional inputs are defined in the building file (e.g., heating/cooling schedules, ventilation rates)
- Outputs depend entirely on the OUTPUTS section in the building file

When generating connections for Type56:
1. Use get_component_io_mapping("Type56", building_file_path="path/to/building.b18") to get complete I/O
2. Parse the returned I/O: identify base 7 inputs + building-file-defined outputs (zones, surfaces, air nodes)
3. Use validate_input_count("Type56", actual_input_count, building_file_path="path/to/building.b18") to verify against TMF spec
4. If building_file_path is not available, fall back to base 7 inputs, but document this limitation for the user

------------------------------------------------------------
Ⅹ. 【[KEY★★★] Connection Format】
INPUTS block format:
```
INPUTS N
source_unit_1,source_output_1  ! SourceUnitName:SourceOutputName ->TargetInputName
source_unit_2,source_output_2  ! EquationUnitName:VariableName ->TargetInputName
0,0                            ! [unconnected] TargetInputName
*** INITIAL INPUT VALUES
init_value_1 init_value_2 ...
```

------------------------------------------------------------
Ⅺ. 【[KEY★★★] Equation Comment Format】
Each equation in "equation_blocks" MUST have a "comment" field following this STRICT format:

**Comment Format**: "description. variable_name [unit]; from [source_unit1,source_output1] output1_name [unit1]; from [source_unit2,source_output2] output2_name [unit2]; to -> [target_unit1,target_input1] input1_name [unit1]"

The comment has THREE parts (separated by ". " and "; "):
1. **Description**: Brief explanation of what the equation calculates
2. **Dependent variable info**: The output variable name and its unit (always present)
3. **Source/Target annotations**:
   - For each [UNIT_ID, OUTPUT_NUMBER] reference in the expression (independent variables), prefix with "from ": "from [source_unit,source_output] output_port_name [unit]"
   - For each component input that receives this equation's output (dependent variable targets), prefix with "to -> ": "to -> [target_unit,target_input] input_port_name [unit]"
   - Use standard port names from TMF files (use get_tmf_spec to look up)

**Example with sources AND targets**:
  "comment": "Control signal based on zone temperature. CTRL [-]; from [1,1] AIR TEMPERATURE [°C]; to -> [8,3] Control signal to pump [-]"

**Example with sources only (no target connections)**:
  "comment": "Temperature difference between zones. T_DIFF [°C]; from [5,1] AIR TEMPERATURE [°C]; from [5,2] SURFACE TEMPERATURE [°C]"

**Example pure constant with targets**:
  "comment": "Zone temperature setpoint. T_SET [°C]; to -> [5,1] Zone Temperature Setpoint [°C]"

**Example pure constant without targets**:
  "comment": "Northern hemisphere flag, 1=Northern, -1=Southern. HEMISPHERE [-]"

------------------------------------------------------------
Ⅻ. 【[KEY★★★] Important Principles】
- MUST call get_tmf_spec for EACH component to get correct INPUT and OUTPUT names
- Use validate_input_count to verify (note: flexible components will show INFO message)
- Learn connection patterns from retrieved examples using two-phase query strategy
- Ensure connections are physically reasonable
- ALL output must be in ENGLISH - no Chinese characters
- Input/Output names MUST match TMF specification EXACTLY
- Support all 4 connection types: Component→Component, Component→Equation, Equation→Component, Equation→Equation

------------------------------------------------------------
ⅩⅢ. 【[KEY★★★] Source and Target Port Physical Quantity MUST Match】

When generating connections, you MUST verify that the physical quantity of the source output port matches the physical quantity of the target input port. Mismatched connections (e.g., temperature output connected to humidity input) will cause WRONG simulation results.

**Physical quantity matching rules:**
1. **Temperature → Temperature**: Source output with unit [C] or [K] can ONLY connect to target input expecting temperature
2. **Humidity → Humidity**: Source output with unit [%] or [-] (humidity ratio) can ONLY connect to target input expecting humidity
3. **Flow rate → Flow rate**: Source output with unit [kg/hr] or [l/hr] can ONLY connect to target input expecting flow rate
4. **Energy/Power → Energy/Power**: Source output with unit [kJ/hr] or [W] can ONLY connect to target input expecting energy/power
5. **Angle → Angle**: Source output with unit [degrees] can ONLY connect to target input expecting angle
6. **Pressure → Pressure**: Source output with unit [atm] or [kPa] can ONLY connect to target input expecting pressure
7. **Control signal → Control signal**: Source output with unit [-] (dimensionless 0/1) can ONLY connect to target input expecting control signal

**FORBIDDEN connections (examples of WRONG connections that MUST be avoided):**
- Temperature output → Humidity input (WRONG!)
- Flow rate output → Temperature input (WRONG!)
- Energy output → Angle input (WRONG!)
- Humidity output → Temperature input (WRONG!)
- Solar radiation output → Temperature input (WRONG!)

**Verification procedure for EVERY connection:**
a) Get the source output name and unit from TMF (via get_component_io_mapping or get_tmf_spec)
b) Get the target input name and unit from TMF
c) Verify the physical quantities are compatible (same physical dimension)
d) If they don't match, find the correct output port that matches the target input's physical quantity

**When using connection patterns from training data (search_connection_patterns / get_connection_pattern):**
- This tool returns port NAMES, not just numbers — use the names to verify physical quantity match
- For flexible-input components (Type56, Type65d, Type534, etc.), port numbers may differ from training data
- ALWAYS match by PORT NAME and physical quantity, NEVER blindly copy port numbers
- If the current component has different parameters than the training data example, the port layout may be different — re-verify each connection

------------------------------------------------------------
ⅩⅣ. 【[KEY★★★] Stop Condition】
- No limit on the maximum number of tool calls; you may call tools as many times as needed to fulfill the prompt requirements
- No strict time requirement for JSON output; output the JSON result once you have completed all prompt requirements (self-judge when ready)
- After generating connections for all components and equations, output the JSON result
- If cannot get I/O mapping, use 0,0 as default connection with "[unconnected] Input N" as name
- Do not repeatedly validate the same component
"""

    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process components and generate connections.

        Args:
            state: Current state containing components with parameters.

        Returns:
            State updates with connections.
        """
        components = state.get("components", [])
        equations = state.get("equations", [])  # Added: receive equations
        task_description = state.get("task_description", "")

        if not components:
            logger.error("No components provided for connection generation")
            return {
                "connections": [],
                "status": "failed",
            }

        self.log_action("Connection generation started", f"Component count: {len(components)}")

        repair_round = self.is_repair_round(state)
        repair_context = self.build_repair_context(state) if repair_round else {}
        repair_context_text = self.build_repair_context_text(repair_context) if repair_round else ""
        prompt_override = self.wrap_prompt_with_repair_instructions(self.system_prompt) if repair_round else None

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

        # Create equation overview if equations exist
        if equations:
            equation_overview = [
                {
                    "unit_number": eq.get("unit_number", "?"),
                    "unit_name": eq.get("unit_name", "Unknown"),
                    "equation_count": eq.get("equation_count", 0),
                }
                for eq in equations
            ]
            equation_overview_text = self._format_state_field_as_json_block(equation_overview)
        else:
            equation_overview_text = "None"

        components_json = self._format_state_field_as_json_block(components)
        equations_json = self._format_state_field_as_json_block(equations) if equations else "[]"

        if repair_round:
            task_message = HumanMessage(content=f"""1. 【Repair Brief】
You are working in a repair round for TRNSYS connection generation.

【【Quick Index — Repair Round — HumanMessage】】
1. 【Repair Brief】 — Previous validation result, errors, failure analysis, earlier chain updates, your editing baseline
2. 【Repair-Round Instructions】 — 7-step: review previous → decide if changes needed → targeted updates → preserve structure → ensure consistency → re-run unit verification on changed connections → same JSON format
3. 【Repair Round - Maximize DCK Tool Usage】 — 4 repair tools: search_connection_patterns, get_connection_pattern, find_dcks_with_components, verify_connection_units
4. 【Original Task】 — Task description for context
5. 【Configured Components Overview】 — Summary of all configured components
6. 【Configured Components Details】 — Full JSON of configured components
7. 【Generated Equations Overview】 — Summary of all equation units
8. 【Generated Equations Details】 — Full JSON of generated equations

{repair_context_text}
------------------------------------------------------------
2. 【Repair-Round Instructions】
1. Review the previous validation result, your previous output, and any updates already made earlier in this repair chain.
2. Decide whether connection-related changes are actually required in your part for this round.
3. If changes are required, update the previous connection result in a targeted way instead of regenerating the full connection design from scratch.
4. If no changes are required, keep the previous connection result logically unchanged and preserve the same output structure.
5. Any update must stay consistent with the current component state, equation state, and physical/system logic.
6. If you change, add, or re-wire ANY connection in this round, you MUST re-run the Step 5 unit & dimension verification (verify_connection_units, batched <=10 pairs/call) over the changed connections and update the top-level unit_verification ledger accordingly. A unit/dimension error introduced during repair is as catastrophic as one in the first round.
7. Return the result in the same JSON structure as the first round (including the unit_verification ledger when connections were checked).

------------------------------------------------------------
3. 【Repair Round - Maximize DCK Tool Usage】
During this repair round, to improve the success rate of connection generation fixes, you should make full use of DCK-related tools to retrieve information about existing validated connection models from the training set:

Available DCK tools for connection generation repair:
- search_connection_patterns: Retrieve connection patterns from training DCKs with RESOLVED PORT NAMES (not just port numbers)
- get_connection_pattern: Query specific connection patterns from training DCKs with PORT NAMES
- find_dcks_with_components: Find training DCKs containing specific component combinations
- verify_connection_units: Re-verify unit & dimension consistency of any connection you changed (MATCH / CONVERT / MISMATCH / UNVERIFIED)

These tools return connection patterns with PORT NAMES resolved from TMF specifications. This ensures you can verify that connections are physically correct (e.g., temperature→temperature, not temperature→humidity). If you can obtain accurate connection patterns from the training set that are highly related to the reported error, it will significantly improve the success rate of your repair.

CRITICAL: When fixing connection errors, always verify that source output and target input have matching physical quantities (see Section ⅩⅢ: Source and Target Port Physical Quantity MUST Match).

------------------------------------------------------------
4. 【Original Task】
{task_description}

------------------------------------------------------------
5. 【Configured Components Overview】
Total {len(components)} components have been configured, including:
{component_overview_text}

------------------------------------------------------------
6. 【Configured Components Details】
{components_json}

------------------------------------------------------------
7. 【Generated Equations Overview】
Total {len(equations)} equation units have been established, including:
{equation_overview_text}

------------------------------------------------------------
8. 【Generated Equations Details】
{equations_json}

Please review the repair context first, decide whether changes are needed, and then produce the updated result.
""")
        else:
            task_message = HumanMessage(content=f"""1. 【Task Instruction】
Generate INPUTS connections for the following TRNSYS components:

【【Quick Index — First Round — HumanMessage】】
1. 【Task Instruction】 — Generate INPUTS connections for all components
2. 【Requirements】 — 10 requirements: workflow, I/O, layers, patterns, visualization, units, initial values, validation, JSON output, equation comments
3. 【Task Description】 — Original task description
4. 【Configured Components Overview】 — Summary of all configured components
5. 【Configured Components Details】 — Full JSON of configured components
6. 【Generated Equations Overview】 — Summary of all equation units
7. 【Generated Equations Details】 — Full JSON of generated equations

------------------------------------------------------------
2. 【Requirements】
1. Strictly follow the 8-step Workflow for connection generation
2. For each component, use get_tmf_spec or get_component_io_mapping to get **complete** I/O information (name, unit, description, default_value, index)
3. Connect in 5 layers: Data Source Layer → Core System Layer → Equation Layer → Control Layer → Visualization/Output Layer
4. Use search_connection_patterns and get_connection_pattern to query high-frequency component pair connection patterns WITH PORT NAMES (query as many pairs as needed). ALWAYS match connections by port NAME and physical quantity, not just port number.
5. Configure visualization/data output connections (not just Type65/Type25, also include Type46/Type155/Type932, etc.)
6. MANDATORY dedicated unit & dimension verification (Step 5): AFTER all connections are fixed, run verify_connection_units (batched, <=10 pairs/call) over EVERY decided connection of ALL 4 types (component->component, component->equation, equation->component, equation->equation). For component ends pass the TMF port; for equation ends pass the unit read from the equation comment. Act per verdict: MATCH keep; CONVERT inject a Unit_Converter_XXX equation (source * factor) and re-route source->equation->target; MISMATCH (different dimension, e.g. temperature->humidity) is a WRONG connection, RE-WIRE it to a port of matching physical quantity (do NOT mask with a conversion equation); UNVERIFIED confirm by port name. Record every checked connection in the top-level unit_verification ledger.
7. Configure INITIAL INPUT VALUES for all INPUTs (distinguish connected/unconnected, prioritize TMF default values; for output/visualization components like Type65/Type25/Type46/Type155/Type932, use source variable names as initial value labels instead of numeric defaults, and replace all spaces with underscores to prevent assembly errors)
8. Check floating inputs and logic deadlocks, use validate_input_count for verification
9. Output JSON format connection configuration (including equation_blocks with both existing and new equations, and initial_values)
10. CRITICAL: Each equation in equation_blocks MUST have a "comment" field with format: "description. variable_name [unit]; from [source_unit,source_output] output_name [unit]; to -> [target_unit,target_input] input_name [unit]". Use "from " prefix for sources and "to -> " prefix for targets. Use get_tmf_spec to look up standard port names for annotations.

------------------------------------------------------------
3. 【Task Description】
{task_description}

------------------------------------------------------------
4. 【Configured Components Overview】
Total {len(components)} components have been configured, including:
{component_overview_text}

------------------------------------------------------------
5. 【Configured Components Details】
{components_json}

------------------------------------------------------------
6. 【Generated Equations Overview】
Total {len(equations)} equation units have been established, including:
{equation_overview_text}

------------------------------------------------------------
7. 【Generated Equations Details】
{equations_json}

Please begin generating connections.
""")

        try:
            # Invoke agent
            response = self.invoke_agent_with_prompt([task_message], prompt_override=prompt_override)

            # Extract result
            result_text = self.extract_result(response)

            # Parse connections
            connections, parsed_data, json_str = self._parse_connections(result_text)

            # Added: parse and merge newly generated equations (avoid duplicates)
            new_equations = self._parse_new_equations(result_text)
            existing_equations = state.get("equations", [])

            # Get existing equation unit_numbers
            existing_unit_numbers = {eq.get("unit_number") for eq in existing_equations}

            # Only add truly new equations (unit_number not already present)
            truly_new_equations = [
                eq for eq in new_equations
                if eq.get("unit_number") not in existing_unit_numbers
            ]

            if truly_new_equations:
                logger.info(f"Adding {len(truly_new_equations)} new equations (filtered out {len(new_equations) - len(truly_new_equations)} duplicates)")
                merged_equations = existing_equations + truly_new_equations
            else:
                merged_equations = existing_equations

            # Merge with components
            updated_components = self._merge_connections(components, connections, merged_equations)

            self.log_action("Connection generation completed", f"Generated connections for {len(connections)} components")

            # CRITICAL: Final check to ensure unit_number is first before returning
            updated_components = self._ensure_unit_number_first(updated_components)
            merged_equations = self._ensure_unit_number_first(merged_equations)

            result = {
                "components": updated_components,
                "connections": connections,
                "equations": merged_equations,  # Added: include newly generated conversion equations
                "status": "connections_generated",
                "current_agent": "connection_generator",
                "next_agent": "layout_designer",
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
                changed = bool(connections) or bool(truly_new_equations)
                repair_type = self.REPAIR_TYPE_DIRECT if state.get("current_agent") == "supervisor" else self.REPAIR_TYPE_ADAPTIVE
                # Extract repair info from LLM output
                repair_summary, repair_detail = self.extract_repair_info_from_output(
                    parsed_data,
                    fallback_summary="Connections regenerated for repair round" if changed else "No connection changes required",
                    fallback_detail=(
                        f"Updated {len(connections)} receiving units and injected {len(truly_new_equations)} new equations"
                        if changed else
                        "Previous round connection output reused"
                    ),
                )
                repair_note = self.build_repair_note(
                    summary=repair_summary,
                    detail=repair_detail,
                    repair_type=repair_type,
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
            logger.error(f"Connection generator error: {e}")
            return {
                "connections": [],
                "status": "failed",
            }

    def _format_state_field_as_json_block(self, value: Any) -> str:
        """
        Render a state field as a unified, model-friendly ```json ... ``` block.

        Goal: make all upstream state fields (``components``, ``equations``,
        ``connections``, ...) share an identical visual and structural form
        inside the human prompt, so the LLM does not need to switch between
        two different presentation styles for upstream data.

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

    def _parse_connections(self, result_text: str) -> tuple[List[Dict[str, Any]], Dict[str, Any], str]:
        """Parse the connection generation result.

        Returns:
            Tuple of (unified list of connection entries, full parsed data including repair info, extracted JSON string)

        Connection entries for ALL receiving units (both components and equations), each with:
        - unit_number, unit_name, type_number (or "EQUATIONS" for equation blocks)
        - inputs: list of input connections
        - initial_values: list of initial values (components only)
        - input_count: number of inputs
        """
        try:
            json_match = re.search(r'```json\s*(.*?)\s*```', result_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1).strip()
                data = json.loads(json_str)
            else:
                json_str = result_text.strip()
                data = json.loads(json_str)

            # Store full data for repair info extraction
            full_data = dict(data)

            # 1. Extract component connections
            comp_connections = data.get("component_connections", data.get("connections", data.get("components", [])))
            if comp_connections:
                logger.info(f"Parsed {len(comp_connections)} component connections")
            else:
                logger.warning(f"No component connections found. Available keys: {list(data.keys())}")
                comp_connections = []

            # CRITICAL: Ensure unit_number is first field immediately after JSON parsing
            comp_connections = self._ensure_unit_number_first(comp_connections)

            # 2. Extract equation connections from equation_blocks
            # Parse both [unit,output] references and variable name references
            eq_blocks = data.get("equation_blocks", [])
            eq_connections = []

            # First pass: build a global variable-to-unit mapping across ALL equation blocks
            # This maps variable_name -> (unit_number, equation_index_1based)
            global_var_map = {}
            for eq_block in eq_blocks:
                block_unit = eq_block.get("unit_number")
                for idx, eq in enumerate(eq_block.get("equations", [])):
                    var_name = eq.get("variable", "")
                    if var_name:
                        global_var_map[var_name] = (block_unit, idx + 1)

            # Math functions and keywords to exclude from variable detection
            math_keywords = {
                'sin', 'cos', 'tan', 'abs', 'max', 'min', 'sqrt', 'exp',
                'log', 'pi', 'GT', 'LT', 'GE', 'LE', 'EQ', 'NE', 'OR',
                'AND', 'NOT', 'MOD', 'INT',
            }

            # Second pass: parse each equation block's inputs
            for eq_block in eq_blocks:
                block_unit = eq_block.get("unit_number")
                eq_inputs = []
                equations_list = eq_block.get("equations", [])

                # Collect all variable names defined in THIS block (for self-reference exclusion)
                block_vars = {eq.get("variable", "") for eq in equations_list}

                for idx, eq in enumerate(equations_list):
                    expression = eq.get("expression", "")
                    variable = eq.get("variable", "")

                    # 2a. Extract [unit,output] references (Component→Equation)
                    unit_ref_pattern = r'\[(\d+),(\d+)\]'
                    unit_matches = re.findall(unit_ref_pattern, expression)

                    for source_unit_str, source_output_str in unit_matches:
                        source_unit = int(source_unit_str)
                        source_output = int(source_output_str)

                        inp_entry = {
                            "index": len(eq_inputs) + 1,
                            "source_type": "component",
                            "source_unit": source_unit,
                            "source_output": source_output,
                            "target_variable": variable,
                            "target_equation_index": idx + 1,
                            "name": f"[{source_unit},{source_output}] ->{variable}"
                        }
                        eq_inputs.append(inp_entry)

                    # 2b. Extract variable name references (Equation→Equation)
                    # Remove [unit,output] patterns first, then find identifiers
                    clean_expr = re.sub(r'\[\d+,\d+\]', '', expression)
                    var_refs = re.findall(r'[A-Za-z_][A-Za-z0-9_]*', clean_expr)

                    for ref_var in var_refs:
                        # Skip: math keywords, self-reference, same-block variables
                        if ref_var in math_keywords:
                            continue
                        if ref_var == variable:
                            continue
                        if ref_var in block_vars:
                            # Same block reference - not a cross-block connection
                            continue
                        # Check if this variable is defined in another equation block
                        if ref_var in global_var_map:
                            src_unit, src_eq_idx = global_var_map[ref_var]
                            if src_unit != block_unit:
                                inp_entry = {
                                    "index": len(eq_inputs) + 1,
                                    "source_type": "equation",
                                    "source_unit": src_unit,
                                    "source_output": src_eq_idx,
                                    "source_variable": ref_var,
                                    "target_variable": variable,
                                    "target_equation_index": idx + 1,
                                    "name": f"Equation_{src_unit}:{ref_var} ->{variable}"
                                }
                                eq_inputs.append(inp_entry)

                # ALWAYS create equation connection entry (even if no inputs)
                eq_conn = {
                    "unit_number": block_unit,
                    "type_number": "EQUATIONS",
                    "unit_name": eq_block.get("unit_name", ""),
                    "inputs": eq_inputs,
                    "initial_values": [],
                    "input_count": len(eq_inputs),
                }
                eq_connections.append(eq_conn)

            logger.info(f"Parsed {len(eq_connections)} equation connection blocks")

            # 3. Merge all connections
            all_connections = comp_connections + eq_connections
            logger.info(f"Total connections: {len(all_connections)} ({len(comp_connections)} components + {len(eq_connections)} equations)")
            return all_connections, full_data, json_str

        except json.JSONDecodeError as e:
            logger.warning(f"Could not parse JSON from connection generator result: {e}")
            logger.debug(f"Result text preview: {result_text[:500]}")
            return [], {}, ""


    def _parse_new_equations(self, result_text: str) -> List[Dict[str, Any]]:
        """Parse new equations generated during connection process (Equation Injection)."""
        try:
            json_match = re.search(r'```json\s*(.*?)\s*```', result_text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(1))
                eq_blocks = data.get("equation_blocks", [])
            else:
                data = json.loads(result_text)
                eq_blocks = data.get("equation_blocks", [])

            # CRITICAL: Ensure unit_number is first field immediately after JSON parsing
            return self._ensure_unit_number_first(eq_blocks)

        except json.JSONDecodeError:
            return []

    def _merge_connections(
        self,
        components: List[Dict[str, Any]],
        connections: List[Dict[str, Any]],
        equations: List[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Merge connection configurations into components and equations.

        Handles all 4 connection types:
        - Component→Component: component input from another component output
        - Equation→Component: component input from equation variable
        - Component→Equation: equation expression references component output
        - Equation→Equation: equation expression references another equation variable
        """
        # Get all defined unit numbers (components + equations)
        defined_units = {comp.get("unit_number") for comp in components}
        if equations:
            for eq in equations:
                eq_unit = eq.get("unit_number")
                if eq_unit is not None:
                    defined_units.add(eq_unit)
        logger.info(f"Defined UNITs (components + equations): {sorted(defined_units)}")

        # Build equation variable to output position mapping
        equation_var_map = {}
        if equations:
            for eq_block in equations:
                unit_num = eq_block.get("unit_number")
                equations_list = eq_block.get("equations", [])
                for idx, eq in enumerate(equations_list, start=1):
                    var_name = eq.get("variable")
                    if var_name:
                        equation_var_map[(unit_num, var_name)] = idx

        # Build lookup by unit_number for all connections (components + equations)
        conn_lookup = {}
        for c in connections:
            unit_num = c.get("unit_number")
            if unit_num is not None:
                conn_lookup[unit_num] = c

        # 1. Merge component connections
        updated = []
        for comp in components:
            unit_num = comp.get("unit_number")

            # Rebuild component dict with unit_number first
            new_comp = {"unit_number": unit_num}

            # Copy all other fields from original component
            for key, value in comp.items():
                if key != "unit_number":
                    new_comp[key] = value

            # Merge connection data if available (won't change position if fields exist)
            if unit_num in conn_lookup:
                conn_data = conn_lookup[unit_num]
                # Only merge if this is a component connection (not equation)
                if conn_data.get("type_number") != "EQUATIONS":
                    validated_inputs = self._validate_connections(
                        conn_data.get("inputs", []),
                        defined_units,
                        equation_var_map,
                    )
                    new_comp["inputs"] = validated_inputs
                    new_comp["initial_values"] = conn_data.get("initial_values", [])
                    # Adjust initial_values count to match inputs
                    if len(new_comp["initial_values"]) != len(validated_inputs):
                        new_comp["initial_values"] = [0] * len(validated_inputs)
                    new_comp["input_count"] = len(validated_inputs)

            updated.append(new_comp)

        # 2. Merge equation connections (Component→Equation, Equation→Equation)
        if equations:
            eq_conn_lookup = {
                unit_num: conn_data
                for unit_num, conn_data in conn_lookup.items()
                if conn_data.get("type_number") == "EQUATIONS"
            }
            if eq_conn_lookup:
                logger.info(f"Merging equation connections for UNITs: {sorted(eq_conn_lookup.keys())}")
                for idx, eq_block in enumerate(equations):
                    eq_unit = eq_block.get("unit_number")
                    if eq_unit in eq_conn_lookup:
                        eq_conn = eq_conn_lookup[eq_unit]

                        # CRITICAL: Rebuild eq_block with unit_number first
                        new_eq_block = {"unit_number": eq_unit}

                        # Copy all other fields from original
                        for key, value in eq_block.items():
                            if key != "unit_number":
                                new_eq_block[key] = value

                        # Store the parsed inputs (won't change position if field exists)
                        new_eq_block["connection_inputs"] = eq_conn.get("inputs", [])

                        logger.info(
                            f"Equation UNIT {eq_unit} ({new_eq_block.get('unit_name', '')}) "
                            f"has {len(new_eq_block['connection_inputs'])} input connections"
                        )

                        # Replace the original eq_block in the list
                        equations[idx] = new_eq_block

        return updated

    def _validate_connections(
        self,
        inputs: List[Dict[str, Any]],
        defined_units: set,
        equation_var_map: Dict = None,
    ) -> List[Dict[str, Any]]:
        """
        Validate connections - ensure source_unit references exist.

        Args:
            inputs: List of input connection dictionaries.
            defined_units: Set of defined unit numbers.
            equation_var_map: Mapping of (unit_number, variable_name) to output position.

        Returns:
            Validated inputs with invalid references replaced by 0,0.
        """
        if equation_var_map is None:
            equation_var_map = {}

        validated = []
        for inp in inputs:
            source_unit = inp.get("source_unit", 0)
            source_output = inp.get("source_output", 0)
            source_type = inp.get("source_type", "component")

            # 0,0 is valid (unconnected)
            if source_unit == 0 and source_output == 0:
                # Normalize the name format for unconnected inputs by extracting the source variable name
                name = inp.get("name", "")
                if "Unconnected" in name or "[unconnected]" not in name:
                    # Extract the source variable name from the existing label (usually after "->")
                    if "->" in name:
                        source_var = name.split("->")[-1].strip()
                        inp["name"] = f"[unconnected] {source_var}"
                validated.append(inp)
                continue

            # Correct the source output for equation-type connections
            if source_type == "equation":
                source_variable = inp.get("source_variable")
                if source_variable and (source_unit, source_variable) in equation_var_map:
                    correct_output = equation_var_map[(source_unit, source_variable)]
                    if source_output != correct_output:
                        logger.info(
                            f"Correcting equation output: Unit {source_unit}, "
                            f"Variable {source_variable}, "
                            f"Output {source_output} -> {correct_output}"
                        )
                        inp["source_output"] = correct_output

            # Check if source_unit is defined
            if source_unit not in defined_units:
                logger.warning(
                    f"Connection references undefined UNIT {source_unit}, "
                    f"replacing with 0,0 (unconnected)"
                )
                inp["source_unit"] = 0
                inp["source_output"] = 0
                # Extract the source variable name
                name = inp.get("name", "")
                if "->" in name:
                    source_var = name.split("->")[-1].strip()
                    inp["name"] = f"[unconnected] {source_var}"
                else:
                    inp["name"] = f"[unconnected] Input {inp.get('index', '')}"

            validated.append(inp)

        return validated
