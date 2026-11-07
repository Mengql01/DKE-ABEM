"""
Equation Tools for TRNSYS DCK Generator.

Provides @tool decorated functions for generating TRNSYS EQUATIONS blocks.
These tools help the equation_generator agent create valid equation syntax.

Tool Categories:
1. Syntax Reference
   - get_equation_syntax_rules: Get TRNSYS EQUATIONS syntax rules

2. Validation
   - validate_equation_syntax: Validate equation syntax
   - validate_equations_block: Validate complete EQUATIONS block

Key Design Principles:
- Based on "TRNSYS_Equation_User_Guide.doc"
- Learn from training set DCK files
- Support the "copy when possible" principle
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


# === EQUATIONS Syntax Rules ===
# Based on TRNSYS documentation and training set analysis

EQUATION_SYNTAX_RULES = """
[TRNSYS EQUATIONS Syntax Rules]

1. [Basic Format]
   The EQUATIONS block is used to define calculated variables and control logic.

   Format:
   EQUATIONS N
   VAR1 = expression1
   VAR2 = expression2
   ...

   Here, N is the number of equations and must match the actual number of equation lines.

2. [Variable Reference Format]
   Reference outputs from other components using:
   [unit_number, output_number]

   Examples:
   T_out = [1, 1]      ! References output 1 of UNIT 1
   Flow = [3, 2]       ! References output 2 of UNIT 3

3. [Supported Operators]
   Arithmetic: +, -, *, /, ^ (power)
   Comparison: GT, GE, LT, LE, EQ, NE (returns 0 or 1)
   Logical: AND, OR, NOT

4. [Built-in Functions]
   Mathematical functions:
   - ABS(x): absolute value
   - SQRT(x): square root
   - EXP(x): exponential
   - LOG(x): natural logarithm
   - LOG10(x): common logarithm
   - SIN(x), COS(x), TAN(x): trigonometric functions (radians)
   - ASIN(x), ACOS(x), ATAN(x): inverse trigonometric functions

   Conditional functions:
   - MAX(a, b): maximum value
   - MIN(a, b): minimum value
   - LT(a, b): returns 1 if a < b, otherwise 0
   - GT(a, b): returns 1 if a > b, otherwise 0
   - LE(a, b): returns 1 if a <= b, otherwise 0
   - GE(a, b): returns 1 if a >= b, otherwise 0
   - EQ(a, b): returns 1 if a == b, otherwise 0
   - NE(a, b): returns 1 if a != b, otherwise 0

   Special functions:
   - OR(a, b): logical OR
   - AND(a, b): logical AND
   - NOT(a): logical NOT
   - EQMOD(a, b): a mod b
   - INT(x): integer part

5. [Conditional Expressions]
   Use GT/LT and similar functions to implement conditional logic.

   Example (temperature control):
   CTRL = GT(T_set, T_actual)  ! Returns 1 when T_set > T_actual

   Example (on/off control):
   ON_OFF = GT(T_in, 20) * LT(T_in, 30)  ! Returns 1 when 20 < T_in < 30

6. [Common Patterns]

   a) Temperature difference calculation:
   DT = [1,1] - [2,1]  ! Temperature difference between two component outputs

   b) Flow control:
   FLOW_CTRL = GT([3,1], 0.5) * 100  ! Output 100 when flow > 0.5

   c) Energy calculation:
   Q = [1,2] * 4.19 * ([1,1] - [1,3])  ! Flow * specific heat * temperature difference

   d) Time control:
   HOUR = EQMOD(TIME, 24)  ! Get the current hour
   DAY_CTRL = GT(HOUR, 8) * LT(HOUR, 18)  ! Returns 1 from 8:00 to 18:00

7. [Important Notes]
   - Variable names are case-sensitive
   - Variable names cannot start with a number
   - Avoid using TRNSYS reserved words
   - Equations are executed in order, and later equations can reference variables defined earlier
   - Each EQUATIONS block has its own UNIT number

8. [Placement in DCK]
   EQUATIONS blocks are typically placed before or between UNIT definitions.
   Each EQUATIONS block must be assigned a unique UNIT number.

   Example:
   UNIT 10 TYPE 65 EQUATIONS
   *$UNIT_NAME Control Equations
   *$POSITION 100 100
   EQUATIONS 3
   T_SET = 22
   CTRL = GT(T_SET, [1,1])
   FLOW = CTRL * 100
"""


# === Equation Validation Patterns ===

# Valid variable name pattern
VALID_VAR_NAME = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')

# Reference pattern [unit, output]
REFERENCE_PATTERN = re.compile(r'\[\s*\d+\s*,\s*\d+\s*\]')

# Built-in functions
BUILTIN_FUNCTIONS = {
    'ABS', 'SQRT', 'EXP', 'LOG', 'LOG10',
    'SIN', 'COS', 'TAN', 'ASIN', 'ACOS', 'ATAN',
    'MAX', 'MIN', 'LT', 'GT', 'LE', 'GE', 'EQ', 'NE',
    'OR', 'AND', 'NOT', 'EQMOD', 'INT',
}

# Reserved words to avoid
RESERVED_WORDS = {
    'UNIT', 'TYPE', 'PARAMETERS', 'INPUTS', 'EQUATIONS',
    'CONSTANTS', 'SIMULATION', 'END', 'TIME', 'STEP',
}


def _validate_variable_name(name: str) -> tuple[bool, str]:
    """Validate a variable name."""
    name = name.strip()

    if not name:
        return False, "Variable name cannot be empty"

    if not VALID_VAR_NAME.match(name):
        return False, f"Variable name '{name}' is invalid (must start with a letter or underscore)"

    if name.upper() in RESERVED_WORDS:
        return False, f"Variable name '{name}' is a TRNSYS reserved word"

    return True, "Valid"


def _validate_expression(expr: str) -> tuple[bool, List[str]]:
    """Validate an equation expression."""
    issues = []

    # Check for balanced parentheses
    if expr.count('(') != expr.count(')'):
        issues.append("Parentheses are not balanced")

    # Check for balanced brackets
    if expr.count('[') != expr.count(']'):
        issues.append("Square brackets are not balanced")

    # Check reference format
    refs = re.findall(r'\[[^\]]+\]', expr)
    for ref in refs:
        if not REFERENCE_PATTERN.match(ref):
            issues.append(f"Invalid reference format: {ref} (expected [unit,output])")

    # Check for common syntax errors
    if '==' in expr:
        issues.append("Using == is invalid; use the EQ() function instead")

    if '!=' in expr:
        issues.append("Using != is invalid; use the NE() function instead")

    if '<=' in expr:
        issues.append("Using <= is invalid; use the LE() function instead")

    if '>=' in expr:
        issues.append("Using >= is invalid; use the GE() function instead")

    # Check for single < or > (might be intentional but warn)
    if re.search(r'(?<![<>])[<>](?![<>=])', expr):
        issues.append("Warning: using < or > directly may cause problems; LT()/GT() is recommended")

    return len(issues) == 0, issues


# === Equation Tools ===

@tool
def get_equation_syntax_rules() -> str:
    """
    Get TRNSYS EQUATIONS syntax rules.

    Returns the complete syntax rules and best practices for EQUATIONS blocks, including:
    - Basic format and structure
    - Variable reference format [unit_number, output_number]
    - Supported operators and functions
    - Conditional expression patterns
    - Common pattern examples
    - Important notes

    Returns:
        Formatted string containing EQUATIONS syntax rules

    Example:
        >>> rules = get_equation_syntax_rules()
        >>> print(rules)
        "[TRNSYS EQUATIONS Syntax Rules]..."
    """
    return EQUATION_SYNTAX_RULES


@tool
def validate_equation_syntax(equation: str) -> str:
    """
    Validate the syntax of a single equation.

    Check whether the equation complies with TRNSYS EQUATIONS syntax rules.

    Args:
        equation: Equation string to validate
            Format: "VAR_NAME = expression"
            Example: "T_OUT = [1,1] + 5"

    Returns:
        Validation result string, including:
        - Whether it is valid
        - List of detected issues
        - Suggested corrections

    Example:
        >>> validate_equation_syntax("T_OUT = [1,1] + 5")
        "✅ Equation syntax is valid"

        >>> validate_equation_syntax("1VAR = [1,1]")
        "❌ Variable name '1VAR' is invalid..."
    """
    try:
        equation = equation.strip()

        if not equation:
            return "❌ Equation is empty"

        # Skip comments
        if equation.startswith('!') or equation.startswith('*'):
            return "✅ This is a comment line, not an equation"

        # Check for assignment
        if '=' not in equation:
            return "❌ Equation is missing the assignment symbol '='"

        # Split into variable and expression
        parts = equation.split('=', 1)
        if len(parts) != 2:
            return "❌ Invalid equation format; expected VAR = expression"

        var_name = parts[0].strip()
        expression = parts[1].strip()

        issues = []

        # Validate variable name
        var_valid, var_msg = _validate_variable_name(var_name)
        if not var_valid:
            issues.append(f"Variable name issue: {var_msg}")

        # Validate expression
        expr_valid, expr_issues = _validate_expression(expression)
        issues.extend(expr_issues)

        # Build result
        if not issues:
            return f"✅ Equation syntax is valid\nVariable: {var_name}\nExpression: {expression}"
        else:
            result = f"❌ Equation has issues:\n"
            result += f"Original equation: {equation}\n"
            result += "Issue list:\n"
            for issue in issues:
                result += f"  - {issue}\n"
            return result

    except Exception as e:
        logger.error(f"Error in validate_equation_syntax: {e}")
        return f"❌ Error during validation: {str(e)}"


@tool
def validate_equations_block(equations_block: str) -> str:
    """
    Validate a complete EQUATIONS block.

    Check the syntax and structure of the entire EQUATIONS block.

    Args:
        equations_block: Complete EQUATIONS block content
            Example:
            '''
            EQUATIONS 3
            T_SET = 22
            CTRL = GT(T_SET, [1,1])
            FLOW = CTRL * 100
            '''

    Returns:
        Validation result string, including:
        - Whether the declared equation count matches
        - Validation result for each equation
        - Variable dependency check
    """
    try:
        lines = equations_block.strip().split('\n')

        if not lines:
            return "❌ EQUATIONS block is empty"

        # Parse header
        header = lines[0].strip()
        header_match = re.match(r'EQUATIONS\s+(\d+)', header, re.IGNORECASE)

        if not header_match:
            return f"❌ Invalid EQUATIONS block header format: {header}\nExpected: EQUATIONS N"

        declared_count = int(header_match.group(1))

        # Parse equations
        equations = []
        defined_vars = set()

        for line in lines[1:]:
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith('!') or line.startswith('*'):
                continue

            if '=' in line:
                equations.append(line)
                var_name = line.split('=')[0].strip()
                defined_vars.add(var_name)

        actual_count = len(equations)

        # Build result
        parts = [f"[EQUATIONS Block Validation Result]\n"]

        # Check count match
        if declared_count != actual_count:
            parts.append(f"❌ Equation count mismatch: declared {declared_count}, actual {actual_count}")
        else:
            parts.append(f"✅ Equation count is correct: {actual_count}")

        # Validate each equation
        parts.append(f"\n[Equation Validation]")
        all_valid = True

        for i, eq in enumerate(equations, 1):
            result = validate_equation_syntax.invoke(eq)
            if "✅" in result:
                parts.append(f"  {i}. ✅ {eq[:50]}...")
            else:
                parts.append(f"  {i}. ❌ {eq[:50]}...")
                parts.append(f"      {result}")
                all_valid = False

        # Check variable dependencies
        parts.append(f"\n[Defined Variables]")
        parts.append(f"  {', '.join(sorted(defined_vars))}")

        # Summary
        parts.append(f"\n[Summary]")
        if declared_count == actual_count and all_valid:
            parts.append("✅ EQUATIONS block validation passed")
        else:
            parts.append("❌ EQUATIONS block has issues and needs correction")

        return "\n".join(parts)

    except Exception as e:
        logger.error(f"Error in validate_equations_block: {e}")
        return f"❌ Error validating EQUATIONS block: {str(e)}"


# === DEPRECATED: search_equation_examples tool removed ===
# This tool was never used in actual testing - all agents use rag_search_equation_examples instead.
# Kept as commented code for reference only.
# Date deprecated: 2026-06-08
# Reason: Testing revealed that this tool was never invoked while rag_search_equation_examples was used multiple times.
#
# @tool
# def search_equation_examples(purpose: str, limit: int = 5) -> str:
#     """
#     Search for equation examples for a specific purpose.
#
#     Search Memory for relevant EQUATIONS block examples.
#     Supports the "copy when possible" principle.
#
#     Args:
#         purpose: Description of the equation purpose
#             Examples: "control logic", "temperature calculation", "flow distribution", "energy statistics", "time control"
#         limit: Maximum number of results to return (default 5)
#
#     Returns:
#         Relevant equation examples, including:
#         - Example code
#         - Purpose description
#         - Usage suggestions
#
#     Note:
#         Memory module has been removed. Returns built-in examples only.
#     """
#     try:
#         # Memory module removed - use built-in examples only
#         examples = _get_builtin_examples(purpose)
#
#         if not examples:
#             return f"No equation examples found for '{purpose}'. Please refer to get_equation_syntax_rules() for syntax guidance."
#
#         parts = [f"[Equation Examples Related to '{purpose}']\n"]
#
#         for i, ex in enumerate(examples[:limit], 1):
#             parts.append(f"\n--- Example {i}: {ex['name']} ---")
#             parts.append(f"Purpose: {ex['description']}")
#             parts.append("Code:")
#             parts.append(ex['code'])
#
#         parts.append("\n[Usage Suggestions]")
#         parts.append("- Update references [unit, output] according to the actual UNIT numbers")
#         parts.append("- Adjust variable names and parameter values based on your requirements")
#         parts.append("- Ensure the number of equations matches the EQUATIONS N declaration")
#
#         return "\n".join(parts)
#
#     except Exception as e:
#         logger.error(f"Error in search_equation_examples: {e}")
#         return f"Error searching equation examples: {str(e)}"


def _get_builtin_examples(purpose: str) -> List[Dict[str, Any]]:
    """Get built-in equation examples based on purpose."""

    all_examples = [
        {
            "name": "Temperature control",
            "keywords": ["temperature", "control", "ctrl"],
            "description": "On/off control based on temperature difference",
            "code": """EQUATIONS 3
T_SET = 22                    ! Setpoint temperature
T_ACTUAL = [1, 1]             ! Actual temperature (from UNIT 1 output 1)
CTRL = GT(T_SET, T_ACTUAL)    ! Turn on when setpoint exceeds actual value"""
        },
        {
            "name": "Flow calculation",
            "keywords": ["flow", "calculation"],
            "description": "Flow calculation based on control signal",
            "code": """EQUATIONS 2
CTRL_SIGNAL = [10, 1]         ! Control signal
FLOW_OUT = CTRL_SIGNAL * 100  ! Flow = control signal * rated flow"""
        },
        {
            "name": "Energy calculation",
            "keywords": ["energy", "heat"],
            "description": "Heating/cooling energy calculation",
            "code": """EQUATIONS 3
FLOW = [1, 2]                 ! Flow rate kg/h
T_IN = [1, 1]                 ! Inlet temperature
T_OUT = [1, 3]                ! Outlet temperature
Q = FLOW * 4.19 * (T_IN - T_OUT)  ! Heat transfer kJ/h"""
        },
        {
            "name": "Time control",
            "keywords": ["time", "schedule"],
            "description": "Time-based control logic",
            "code": """EQUATIONS 3
HOUR = EQMOD(TIME, 24)        ! Current hour
DAY_ON = GT(HOUR, 8) * LT(HOUR, 18)  ! 1 from 8:00 to 18:00
NIGHT_ON = NOT(DAY_ON)        ! 1 at night"""
        },
        {
            "name": "Conditional logic",
            "keywords": ["condition", "logic", "if"],
            "description": "Combined multi-condition control",
            "code": """EQUATIONS 4
T_ZONE = [56, 1]              ! Zone temperature
T_SET_COOL = 26               ! Cooling setpoint
T_SET_HEAT = 20               ! Heating setpoint
COOL_ON = GT(T_ZONE, T_SET_COOL)  ! Cooling required
HEAT_ON = LT(T_ZONE, T_SET_HEAT)  ! Heating required"""
        },
        {
            "name": "Min/max limiting",
            "keywords": ["max", "min", "limit"],
            "description": "Value limiting and selection",
            "code": """EQUATIONS 3
RAW_VALUE = [1, 1]            ! Raw value
LIMITED = MAX(0, MIN(100, RAW_VALUE))  ! Clamp to 0-100
SELECTED = MAX([1,1], [2,1])  ! Select the larger value"""
        },
        {
            "name": "Pump control",
            "keywords": ["pump"],
            "description": "Pump start/stop and flow control",
            "code": """EQUATIONS 4
DEMAND = [10, 1]              ! Demand signal
PUMP_ON = GT(DEMAND, 0.1)     ! Start pump when demand > 0.1
FLOW_RATE = PUMP_ON * DEMAND * 1000  ! Flow rate kg/h
POWER = PUMP_ON * 5           ! Power kW"""
        },
        {
            "name": "Chiller control",
            "keywords": ["chiller"],
            "description": "Chiller control logic",
            "code": """EQUATIONS 5
T_CHW_OUT = [927, 1]          ! Chilled water outlet temperature
T_SET = 7                     ! Setpoint temperature
LOAD = [927, 3]               ! Load ratio
CHILLER_ON = GT(LOAD, 0.1)    ! Run when load > 10%
CTRL_SIGNAL = CHILLER_ON * MIN(1, LOAD)  ! Control signal"""
        },
    ]

    # Filter by purpose
    purpose_lower = purpose.lower()
    matched = []

    for ex in all_examples:
        if any(kw in purpose_lower for kw in ex["keywords"]):
            matched.append(ex)

    # If no match, return all examples
    if not matched:
        matched = all_examples

    return matched


# === Tool Collection Functions ===

def get_equation_tools() -> list:
    """
    Get all equation tools (3).

    Returns:
        List of @tool decorated functions.
    """
    return [
        get_equation_syntax_rules,
        validate_equation_syntax,
        validate_equations_block,
        # search_equation_examples,  # DEPRECATED: Never used in testing
    ]


def get_equation_tools_for_agent(agent_role: str) -> list:
    """
    Get equation tools for a specific agent role.

    Args:
        agent_role: Agent role name

    Returns:
        List of equation tools appropriate for that role
    """
    role_tools = {
        "planner": [],
        "component_selector": [],
        "parameter_configurator": [],
        "connection_generator": [validate_equation_syntax],
        "equation_generator": [
            get_equation_syntax_rules,
            validate_equation_syntax,
            validate_equations_block,
            # search_equation_examples,  # DEPRECATED: Never used in testing
        ],
        "layout_designer": [],
        "validator": [
            validate_equation_syntax,
            validate_equations_block,
        ],
        "supervisor": get_equation_tools(),
    }

    return role_tools.get(agent_role, [])
