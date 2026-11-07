"""
DCK Format Checker for TRNSYS DCK Generator.

This module provides strict format validation for DCK files.
Validation rules are derived from the 176 training set DCK files and TMF specifications.

Key Features:
1. VERSION validation
2. SIMULATION block validation
3. UNIT block format validation
4. Parameter count validation (against TMF specs)
5. Input count validation (against TMF specs)
6. Connection reference validation
7. END marker validation

Design Principles:
- Validation rules are based on training set analysis
- TMF data is used for parameter/input count verification
- Provides detailed error messages with suggestions
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class ErrorSeverity(str, Enum):
    """Severity levels for format errors."""

    ERROR = "error"  # Will cause TrnEXE64 to fail
    WARNING = "warning"  # May cause issues
    INFO = "info"  # Informational only


class ErrorType(str, Enum):
    """Types of format errors."""

    VERSION_MISSING = "version_missing"
    VERSION_INVALID = "version_invalid"
    SIMULATION_MISSING = "simulation_missing"
    SIMULATION_INVALID = "simulation_invalid"
    CONSTANTS_MISMATCH = "constants_mismatch"
    UNIT_INVALID = "unit_invalid"
    UNIT_DUPLICATE = "unit_duplicate"
    PARAMETERS_MISMATCH = "parameters_mismatch"
    INPUTS_MISMATCH = "inputs_mismatch"
    INITIAL_VALUES_MISMATCH = "initial_values_mismatch"
    CONNECTION_INVALID = "connection_invalid"
    CONNECTION_REFERENCE = "connection_reference"
    EQUATIONS_INVALID = "equations_invalid"
    END_MISSING = "end_missing"
    SYNTAX_ERROR = "syntax_error"


@dataclass
class FormatError:
    """Represents a format validation error."""

    error_type: ErrorType
    message: str
    severity: ErrorSeverity = ErrorSeverity.ERROR
    line_number: Optional[int] = None
    section: Optional[str] = None
    unit_number: Optional[int] = None
    suggestion: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "error_type": self.error_type.value,
            "message": self.message,
            "severity": self.severity.value,
            "line_number": self.line_number,
            "section": self.section,
            "unit_number": self.unit_number,
            "suggestion": self.suggestion,
        }

    def __str__(self) -> str:
        """String representation."""
        parts = [f"[{self.severity.value.upper()}] {self.error_type.value}: {self.message}"]
        if self.line_number:
            parts.append(f" (line {self.line_number})")
        if self.unit_number:
            parts.append(f" (UNIT {self.unit_number})")
        if self.suggestion:
            parts.append(f" - Suggestion: {self.suggestion}")
        return "".join(parts)


@dataclass
class ValidationResult:
    """Result of DCK format validation."""

    is_valid: bool
    errors: List[FormatError] = field(default_factory=list)
    warnings: List[FormatError] = field(default_factory=list)
    info: List[FormatError] = field(default_factory=list)

    @property
    def error_count(self) -> int:
        """Number of errors."""
        return len(self.errors)

    @property
    def warning_count(self) -> int:
        """Number of warnings."""
        return len(self.warnings)

    @property
    def all_issues(self) -> List[FormatError]:
        """All issues (errors + warnings + info)."""
        return self.errors + self.warnings + self.info

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_valid": self.is_valid,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "errors": [e.to_dict() for e in self.errors],
            "warnings": [w.to_dict() for w in self.warnings],
            "info": [i.to_dict() for i in self.info],
        }


class TMFDataLoader:
    """Loader for TMF component specification data."""

    def __init__(self, tmf_data_path: Path = None):
        """
        Initialize the TMF data loader.

        Args:
            tmf_data_path: Path to TMF parsed JSON file.
        """
        self._tmf_data_path = tmf_data_path
        self._tmf_specs: Dict[str, Dict[str, Any]] = {}
        self._loaded = False

    def _ensure_loaded(self):
        """Ensure TMF data is loaded."""
        if not self._loaded:
            self._load_tmf_data()
            self._loaded = True

    def _load_tmf_data(self):
        """Load TMF specification data."""
        if self._tmf_data_path is None:
            try:
                from ..core.config import get_settings
                settings = get_settings()
                self._tmf_data_path = settings.tmf_parsed_path
            except Exception as e:
                logger.warning(f"Could not get TMF data path from config: {e}")
                return

        if not self._tmf_data_path or not self._tmf_data_path.exists():
            logger.warning(f"TMF data path not found: {self._tmf_data_path}")
            return

        try:
            with open(self._tmf_data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Handle different JSON structures
            if isinstance(data, dict):
                if "results" in data:
                    tmf_list = data["results"]
                else:
                    tmf_list = list(data.values())
            elif isinstance(data, list):
                tmf_list = data
            else:
                logger.warning(f"Unexpected TMF data format")
                return

            # Index by type number
            for tmf in tmf_list:
                if isinstance(tmf, dict):
                    type_num = tmf.get("type_number") or tmf.get("type") or tmf.get("Type")
                    if type_num:
                        # Normalize type number
                        type_key = str(type_num).replace("Type", "").strip()
                        self._tmf_specs[type_key] = tmf

            logger.info(f"Loaded TMF specs for {len(self._tmf_specs)} component types")

        except Exception as e:
            logger.error(f"Error loading TMF data: {e}")

    def get_spec(self, type_id: str) -> Optional[Dict[str, Any]]:
        """
        Get TMF specification for a type.

        Args:
            type_id: Type ID (e.g., "15", "56", "Type15").

        Returns:
            TMF specification dictionary or None.
        """
        self._ensure_loaded()

        # Normalize type_id
        type_key = str(type_id).replace("Type", "").replace("type", "").strip()

        return self._tmf_specs.get(type_key)

    def get_parameter_count(self, type_id: str) -> Optional[int]:
        """Get expected parameter count for a type."""
        spec = self.get_spec(type_id)
        if spec:
            # Try different field names
            for field_name in ["parameter_count", "num_parameters", "parameters_count", "n_parameters"]:
                if field_name in spec:
                    return int(spec[field_name])

            # Try to count from parameters list
            params = spec.get("parameters", [])
            if isinstance(params, list):
                return len(params)

        return None

    def get_input_count(self, type_id: str) -> Optional[int]:
        """Get expected input count for a type."""
        spec = self.get_spec(type_id)
        if spec:
            # Try different field names
            for field_name in ["input_count", "num_inputs", "inputs_count", "n_inputs"]:
                if field_name in spec:
                    return int(spec[field_name])

            # Try to count from inputs list
            inputs = spec.get("inputs", [])
            if isinstance(inputs, list):
                return len(inputs)

        return None

    def get_output_count(self, type_id: str) -> Optional[int]:
        """Get expected output count for a type."""
        spec = self.get_spec(type_id)
        if spec:
            # Try different field names
            for field_name in ["output_count", "num_outputs", "outputs_count", "n_outputs"]:
                if field_name in spec:
                    return int(spec[field_name])

            # Try to count from outputs list
            outputs = spec.get("outputs", [])
            if isinstance(outputs, list):
                return len(outputs)

        return None


class DCKFormatChecker:
    """
    DCK Format Checker.

    Validates DCK file format based on training set rules and TMF specifications.
    """

    # Regex patterns for DCK parsing
    VERSION_PATTERN = re.compile(r'^VERSION\s+(\d+)', re.MULTILINE)
    SIMULATION_PATTERN = re.compile(r'^SIMULATION\s+(\S+)\s+(\S+)\s+(\S+)', re.MULTILINE)
    CONSTANTS_PATTERN = re.compile(r'^CONSTANTS\s+(\d+)', re.MULTILINE)
    # UNIT_PATTERN: type_number can be digits, or digits with suffix (e.g., 15, 15-6, 65d)
    UNIT_PATTERN = re.compile(r'^UNIT\s+(\d+)\s+TYPE\s+(\d+[\w\-]*)\s+(\S+)', re.MULTILINE)
    PARAMETERS_PATTERN = re.compile(r'^PARAMETERS\s+(\d+)', re.MULTILINE)
    INPUTS_PATTERN = re.compile(r'^INPUTS\s+(\d+)', re.MULTILINE)
    EQUATIONS_PATTERN = re.compile(r'^EQUATIONS\s+(\d+)', re.MULTILINE)
    END_PATTERN = re.compile(r'^END\s*$', re.MULTILINE)
    CONNECTION_PATTERN = re.compile(r'^(\d+),(\d+)', re.MULTILINE)

    def __init__(self, tmf_loader: TMFDataLoader = None):
        """
        Initialize the format checker.

        Args:
            tmf_loader: TMF data loader for specification validation.
        """
        self.tmf_loader = tmf_loader or TMFDataLoader()

    def validate_complete_dck(self, dck_content: str) -> ValidationResult:
        """
        Validate a complete DCK file.

        Args:
            dck_content: The DCK file content.

        Returns:
            ValidationResult with all errors and warnings.
        """
        all_errors = []

        # Run all checks
        all_errors.extend(self.check_version(dck_content))
        all_errors.extend(self.check_simulation_block(dck_content))
        all_errors.extend(self.check_constants_block(dck_content))
        all_errors.extend(self.check_all_units(dck_content))
        all_errors.extend(self.check_connections(dck_content))
        all_errors.extend(self.check_end_marker(dck_content))

        # Categorize by severity
        errors = [e for e in all_errors if e.severity == ErrorSeverity.ERROR]
        warnings = [e for e in all_errors if e.severity == ErrorSeverity.WARNING]
        info = [e for e in all_errors if e.severity == ErrorSeverity.INFO]

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            info=info,
        )

    def check_version(self, dck_content: str) -> List[FormatError]:
        """Check VERSION declaration."""
        errors = []

        match = self.VERSION_PATTERN.search(dck_content)
        if not match:
            errors.append(FormatError(
                error_type=ErrorType.VERSION_MISSING,
                message="VERSION declaration not found",
                severity=ErrorSeverity.ERROR,
                suggestion="Add 'VERSION 18' at the beginning of the file",
            ))
        else:
            version = int(match.group(1))
            if version != 18:
                errors.append(FormatError(
                    error_type=ErrorType.VERSION_INVALID,
                    message=f"VERSION {version} is not supported, expected VERSION 18",
                    severity=ErrorSeverity.WARNING,
                    suggestion="Use 'VERSION 18' for TRNSYS 18",
                ))

        return errors

    def check_simulation_block(self, dck_content: str) -> List[FormatError]:
        """Check SIMULATION block."""
        errors = []

        match = self.SIMULATION_PATTERN.search(dck_content)
        if not match:
            errors.append(FormatError(
                error_type=ErrorType.SIMULATION_MISSING,
                message="SIMULATION declaration not found",
                severity=ErrorSeverity.ERROR,
                section="SIMULATION",
                suggestion="Add 'SIMULATION START STOP STEP' after CONSTANTS block",
            ))
        else:
            start, stop, step = match.groups()

            # Validate that these are valid references or numbers
            try:
                # Check if they're numbers or variable references
                for name, value in [("START", start), ("STOP", stop), ("STEP", step)]:
                    if not self._is_valid_value(value):
                        errors.append(FormatError(
                            error_type=ErrorType.SIMULATION_INVALID,
                            message=f"Invalid {name} value: {value}",
                            severity=ErrorSeverity.WARNING,
                            section="SIMULATION",
                        ))
            except Exception:
                pass

        return errors

    def check_constants_block(self, dck_content: str) -> List[FormatError]:
        """Check CONSTANTS block."""
        errors = []

        match = self.CONSTANTS_PATTERN.search(dck_content)
        if match:
            declared_count = int(match.group(1))

            # Count actual constant definitions after CONSTANTS N
            start_pos = match.end()
            # Find the next major keyword
            next_keyword = re.search(
                r'^(SIMULATION|TOLERANCES|LIMITS|DFQ|WIDTH|LIST|SOLVER|UNIT|EQUATIONS|END)',
                dck_content[start_pos:],
                re.MULTILINE
            )

            if next_keyword:
                constants_section = dck_content[start_pos:start_pos + next_keyword.start()]
            else:
                constants_section = dck_content[start_pos:start_pos + 500]

            # Count lines that look like constant definitions (NAME=VALUE)
            const_lines = re.findall(r'^(\w+)\s*=\s*(.+)$', constants_section, re.MULTILINE)
            actual_count = len(const_lines)

            if actual_count != declared_count:
                errors.append(FormatError(
                    error_type=ErrorType.CONSTANTS_MISMATCH,
                    message=f"CONSTANTS declares {declared_count} but found {actual_count} definitions",
                    severity=ErrorSeverity.WARNING,
                    section="CONSTANTS",
                    suggestion=f"Change 'CONSTANTS {declared_count}' to 'CONSTANTS {actual_count}'",
                ))

        return errors

    def check_all_units(self, dck_content: str) -> List[FormatError]:
        """Check all UNIT blocks."""
        errors = []
        seen_units: Set[int] = set()

        # Find all UNIT declarations
        for match in self.UNIT_PATTERN.finditer(dck_content):
            unit_num = int(match.group(1))
            type_num = match.group(2)
            unit_name = match.group(3)

            # Check for duplicate unit numbers
            if unit_num in seen_units:
                errors.append(FormatError(
                    error_type=ErrorType.UNIT_DUPLICATE,
                    message=f"Duplicate UNIT number: {unit_num}",
                    severity=ErrorSeverity.ERROR,
                    unit_number=unit_num,
                    suggestion="Each UNIT must have a unique number",
                ))
            seen_units.add(unit_num)

            # Extract the UNIT block content
            unit_start = match.start()
            unit_end = self._find_unit_end(dck_content, unit_start)
            unit_content = dck_content[unit_start:unit_end]

            # Check UNIT block format
            unit_errors = self.check_unit_block(unit_content, type_num, unit_num)
            errors.extend(unit_errors)

        return errors

    def check_unit_block(
        self,
        unit_content: str,
        type_id: str,
        unit_number: int,
    ) -> List[FormatError]:
        """
        Check a single UNIT block.

        Args:
            unit_content: The UNIT block content.
            type_id: The Type ID.
            unit_number: The UNIT number.

        Returns:
            List of format errors.
        """
        errors = []

        # Skip TYPE 65 (EQUATIONS) - different format
        if type_id == "65":
            return self._check_equations_unit(unit_content, unit_number)

        # Check PARAMETERS
        param_match = self.PARAMETERS_PATTERN.search(unit_content)
        if param_match:
            declared_params = int(param_match.group(1))

            # Count actual parameter lines
            param_start = param_match.end()
            param_section = self._extract_section_content(unit_content, param_start, ["INPUTS", "UNIT", "END"])
            actual_params = self._count_value_lines(param_section)

            if actual_params != declared_params:
                errors.append(FormatError(
                    error_type=ErrorType.PARAMETERS_MISMATCH,
                    message=f"PARAMETERS declares {declared_params} but found {actual_params} values",
                    severity=ErrorSeverity.ERROR,
                    unit_number=unit_number,
                    section="PARAMETERS",
                    suggestion=f"Ensure exactly {declared_params} parameter values are provides",
                ))

            # Check against TMF spec
            expected_params = self.tmf_loader.get_parameter_count(type_id)
            if expected_params is not None and declared_params != expected_params:
                errors.append(FormatError(
                    error_type=ErrorType.PARAMETERS_MISMATCH,
                    message=f"Type{type_id} expects {expected_params} parameters but {declared_params} declared",
                    severity=ErrorSeverity.WARNING,
                    unit_number=unit_number,
                    section="PARAMETERS",
                    suggestion=f"Check TMF specification for Type{type_id}",
                ))

        # Check INPUTS
        input_match = self.INPUTS_PATTERN.search(unit_content)
        if input_match:
            declared_inputs = int(input_match.group(1))

            # Count actual input lines
            input_start = input_match.end()
            input_section = self._extract_section_content(
                unit_content, input_start, ["*** INITIAL", "UNIT", "END", "*---"]
            )
            actual_inputs = self._count_connection_lines(input_section)

            if actual_inputs != declared_inputs:
                errors.append(FormatError(
                    error_type=ErrorType.INPUTS_MISMATCH,
                    message=f"INPUTS declares {declared_inputs} but found {actual_inputs} connections",
                    severity=ErrorSeverity.ERROR,
                    unit_number=unit_number,
                    section="INPUTS",
                    suggestion=f"Ensure exactly {declared_inputs} input connections are provides",
                ))

            # Check initial values
            init_match = re.search(r'\*\*\* INITIAL INPUT VALUES\s*\n(.+)', unit_content)
            if init_match:
                init_line = init_match.group(1).strip()
                init_values = init_line.split()
                if len(init_values) != declared_inputs:
                    errors.append(FormatError(
                        error_type=ErrorType.INITIAL_VALUES_MISMATCH,
                        message=f"Expected {declared_inputs} initial values but found {len(init_values)}",
                        severity=ErrorSeverity.ERROR,
                        unit_number=unit_number,
                        section="INITIAL INPUT VALUES",
                        suggestion=f"Provide exactly {declared_inputs} initial values",
                    ))

            # Check against TMF spec
            expected_inputs = self.tmf_loader.get_input_count(type_id)
            if expected_inputs is not None and declared_inputs != expected_inputs:
                errors.append(FormatError(
                    error_type=ErrorType.INPUTS_MISMATCH,
                    message=f"Type{type_id} expects {expected_inputs} inputs but {declared_inputs} declared",
                    severity=ErrorSeverity.WARNING,
                    unit_number=unit_number,
                    section="INPUTS",
                    suggestion=f"Check TMF specification for Type{type_id}",
                ))

        return errors

    def _check_equations_unit(self, unit_content: str, unit_number: int) -> List[FormatError]:
        """Check an EQUATIONS unit (TYPE 65)."""
        errors = []

        eq_match = self.EQUATIONS_PATTERN.search(unit_content)
        if eq_match:
            declared_eqs = int(eq_match.group(1))

            # Count actual equation lines
            eq_start = eq_match.end()
            eq_section = self._extract_section_content(unit_content, eq_start, ["UNIT", "END", "*---"])

            # Count non-empty, non-comment lines
            actual_eqs = 0
            for line in eq_section.split('\n'):
                line = line.strip()
                if line and not line.startswith('*') and '=' in line:
                    actual_eqs += 1

            if actual_eqs != declared_eqs:
                errors.append(FormatError(
                    error_type=ErrorType.EQUATIONS_INVALID,
                    message=f"EQUATIONS declares {declared_eqs} but found {actual_eqs} equations",
                    severity=ErrorSeverity.ERROR,
                    unit_number=unit_number,
                    section="EQUATIONS",
                    suggestion=f"Ensure exactly {declared_eqs} equations are provides",
                ))

        return errors

    def check_connections(self, dck_content: str) -> List[FormatError]:
        """Check connection references."""
        errors = []

        # Get all defined unit numbers
        defined_units = set()
        for match in self.UNIT_PATTERN.finditer(dck_content):
            defined_units.add(int(match.group(1)))

        # Find all connection references (unit,output format)
        for match in self.CONNECTION_PATTERN.finditer(dck_content):
            src_unit = int(match.group(1))
            src_output = int(match.group(2))

            # Skip 0,0 (unconnected)
            if src_unit == 0 and src_output == 0:
                continue

            # Check if source unit exists
            if src_unit not in defined_units:
                # Find line number
                line_num = dck_content[:match.start()].count('\n') + 1

                errors.append(FormatError(
                    error_type=ErrorType.CONNECTION_REFERENCE,
                    message=f"Connection references undefined UNIT {src_unit}",
                    severity=ErrorSeverity.ERROR,
                    line_number=line_num,
                    section="INPUTS",
                    suggestion=f"Ensure UNIT {src_unit} is defined before referencing it",
                ))

        return errors

    def check_end_marker(self, dck_content: str) -> List[FormatError]:
        """Check END marker."""
        errors = []

        if not self.END_PATTERN.search(dck_content):
            errors.append(FormatError(
                error_type=ErrorType.END_MISSING,
                message="END marker not found",
                severity=ErrorSeverity.ERROR,
                suggestion="Add 'END' at the end of the DCK file (before LINK_STYLE if present)",
            ))

        return errors

    def check_equations(self, equations: List[str]) -> List[FormatError]:
        """
        Check equation syntax.

        Args:
            equations: List of equation strings.

        Returns:
            List of format errors.
        """
        errors = []

        for i, eq in enumerate(equations, 1):
            eq = eq.strip()
            if not eq or eq.startswith('*'):
                continue

            # Check for assignment operator
            if '=' not in eq:
                errors.append(FormatError(
                    error_type=ErrorType.EQUATIONS_INVALID,
                    message=f"Equation {i} missing '=' operator: {eq[:50]}",
                    severity=ErrorSeverity.ERROR,
                    section="EQUATIONS",
                ))
                continue

            # Check variable name (left side)
            parts = eq.split('=', 1)
            var_name = parts[0].strip()

            if not re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', var_name):
                errors.append(FormatError(
                    error_type=ErrorType.EQUATIONS_INVALID,
                    message=f"Invalid variable name in equation {i}: {var_name}",
                    severity=ErrorSeverity.ERROR,
                    section="EQUATIONS",
                    suggestion="Variable names must start with a letter and contain only letters, numbers, and underscores",
                ))

            # Check for balanced brackets
            expression = parts[1].strip()
            if expression.count('[') != expression.count(']'):
                errors.append(FormatError(
                    error_type=ErrorType.EQUATIONS_INVALID,
                    message=f"Unbalanced brackets in equation {i}",
                    severity=ErrorSeverity.ERROR,
                    section="EQUATIONS",
                ))

            if expression.count('(') != expression.count(')'):
                errors.append(FormatError(
                    error_type=ErrorType.EQUATIONS_INVALID,
                    message=f"Unbalanced parentheses in equation {i}",
                    severity=ErrorSeverity.ERROR,
                    section="EQUATIONS",
                ))

        return errors

    # === Helper Methods ===

    def _is_valid_value(self, value: str) -> bool:
        """Check if a value is valid (number or variable reference)."""
        # Try as number
        try:
            float(value)
            return True
        except ValueError:
            pass

        # Check as variable reference
        if re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', value):
            return True

        return False

    def _find_unit_end(self, content: str, start: int) -> int:
        """Find the end of a UNIT block."""
        # Look for next UNIT, END, or separator line
        remaining = content[start:]

        # Find next UNIT declaration
        next_unit = re.search(r'\nUNIT\s+\d+\s+TYPE', remaining)
        if next_unit:
            return start + next_unit.start()

        # Find END
        end_match = re.search(r'\nEND\s*$', remaining, re.MULTILINE)
        if end_match:
            return start + end_match.start()

        # Find separator line
        sep_match = re.search(r'\n\*-{10,}', remaining)
        if sep_match:
            return start + sep_match.end()

        return len(content)

    def _extract_section_content(
        self,
        content: str,
        start: int,
        end_markers: List[str],
    ) -> str:
        """Extract content between start position and first end marker."""
        remaining = content[start:]

        min_end = len(remaining)
        for marker in end_markers:
            match = re.search(re.escape(marker), remaining, re.IGNORECASE)
            if match and match.start() < min_end:
                min_end = match.start()

        return remaining[:min_end]

    def _count_value_lines(self, section: str) -> int:
        """Count lines that contain values (non-empty, non-comment)."""
        count = 0
        for line in section.split('\n'):
            line = line.strip()
            if line and not line.startswith('*'):
                # Check if line contains a value (number or variable)
                # Remove comments
                if '!' in line:
                    line = line.split('!')[0].strip()
                if line:
                    count += 1
        return count

    def _count_connection_lines(self, section: str) -> int:
        """Count lines that contain connection definitions (unit,output format)."""
        count = 0
        for line in section.split('\n'):
            line = line.strip()
            if line and not line.startswith('*'):
                # Check for connection format
                if re.match(r'^\d+\s*,\s*\d+', line):
                    count += 1
        return count


# Global format checker instance
_format_checker: Optional[DCKFormatChecker] = None


def get_format_checker() -> DCKFormatChecker:
    """Get the global format checker instance."""
    global _format_checker
    if _format_checker is None:
        _format_checker = DCKFormatChecker()
    return _format_checker


def reset_format_checker():
    """Reset the global format checker (for testing)."""
    global _format_checker
    _format_checker = None


def validate_dck(dck_content: str) -> ValidationResult:
    """
    Convenience function to validate DCK content.

    Args:
        dck_content: The DCK file content.

    Returns:
        ValidationResult.
    """
    checker = get_format_checker()
    return checker.validate_complete_dck(dck_content)
