"""
Evaluation Metrics for TRNSYS DCK Generator.

This module provides 5 evaluation metrics for assessing DCK generation quality:
1. SC (Syntax Correctness) - DCK syntax validation
2. CA (Component Accuracy) - Component selection accuracy vs reference
3. CC (Connection Completeness) - Connection completeness vs reference
4. PR (Parameter Reasonableness) - Parameter validity vs TMF specs
5. ESR (Execution Success Rate) - TrnEXE64.exe execution success (MOST IMPORTANT)

Design Principles:
- ESR is the most important metric (must reach 100% for Proposed method)
- Metrics are calculated by comparing generated DCK with reference DCK
- TMF data is used for parameter validation
- No hardcoded rules - uses existing validation modules
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class MetricType(str, Enum):
    """Types of evaluation metrics."""

    SC = "syntax_correctness"  # Syntax Correctness
    CA = "component_accuracy"  # Component Accuracy
    CC = "connection_completeness"  # Connection Completeness
    PR = "parameter_reasonableness"  # Parameter Reasonableness
    ESR = "execution_success_rate"  # Execution Success Rate (MOST IMPORTANT)


@dataclass
class MetricResult:
    """Result of a single metric calculation."""

    metric_type: MetricType
    value: float  # 0.0 to 1.0
    passed: bool  # Whether the metric passes threshold
    details: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metric_type": self.metric_type.value,
            "value": self.value,
            "passed": self.passed,
            "details": self.details,
            "errors": self.errors,
        }


@dataclass
class EvaluationResult:
    """Complete evaluation result for a single test case."""

    task_id: str
    success: bool  # Overall success (ESR passed)

    # Individual metrics
    syntax_correctness: MetricResult = None
    component_accuracy: MetricResult = None
    connection_completeness: MetricResult = None
    parameter_reasonableness: MetricResult = None
    execution_success: MetricResult = None

    # Additional info
    generated_dck: str = ""
    reference_dck: str = ""
    generation_time: float = 0.0
    validation_log: str = ""

    @property
    def all_metrics(self) -> Dict[str, MetricResult]:
        """Get all metrics as dictionary."""
        return {
            MetricType.SC.value: self.syntax_correctness,
            MetricType.CA.value: self.component_accuracy,
            MetricType.CC.value: self.connection_completeness,
            MetricType.PR.value: self.parameter_reasonableness,
            MetricType.ESR.value: self.execution_success,
        }

    @property
    def esr_passed(self) -> bool:
        """Check if ESR (execution success) passed."""
        return self.execution_success is not None and self.execution_success.passed

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_id": self.task_id,
            "success": self.success,
            "metrics": {
                "SC": self.syntax_correctness.to_dict() if self.syntax_correctness else None,
                "CA": self.component_accuracy.to_dict() if self.component_accuracy else None,
                "CC": self.connection_completeness.to_dict() if self.connection_completeness else None,
                "PR": self.parameter_reasonableness.to_dict() if self.parameter_reasonableness else None,
                "ESR": self.execution_success.to_dict() if self.execution_success else None,
            },
            "generation_time": self.generation_time,
        }


@dataclass
class AggregatedMetrics:
    """Aggregated metrics across multiple test cases."""

    total_cases: int
    passed_cases: int

    # Average metrics
    avg_syntax_correctness: float = 0.0
    avg_component_accuracy: float = 0.0
    avg_connection_completeness: float = 0.0
    avg_parameter_reasonableness: float = 0.0

    # ESR is calculated as passed_cases / total_cases
    execution_success_rate: float = 0.0

    # Individual results
    results: List[EvaluationResult] = field(default_factory=list)

    @property
    def esr_100_percent(self) -> bool:
        """Check if ESR is 100%."""
        return self.execution_success_rate == 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_cases": self.total_cases,
            "passed_cases": self.passed_cases,
            "metrics": {
                "SC": self.avg_syntax_correctness,
                "CA": self.avg_component_accuracy,
                "CC": self.avg_connection_completeness,
                "PR": self.avg_parameter_reasonableness,
                "ESR": self.execution_success_rate,
            },
            "esr_100_percent": self.esr_100_percent,
        }


class DCKParser:
    """Parser for extracting information from DCK files."""

    # Regex patterns for DCK parsing
    UNIT_PATTERN = re.compile(
        r'UNIT\s+(\d+)\s+TYPE\s+(\d+)\s+(\S+)',
        re.IGNORECASE
    )
    PARAMETERS_PATTERN = re.compile(
        r'PARAMETERS\s+(\d+)',
        re.IGNORECASE
    )
    INPUTS_PATTERN = re.compile(
        r'INPUTS\s+(\d+)',
        re.IGNORECASE
    )
    CONNECTION_PATTERN = re.compile(
        r'(\d+),(\d+)',
    )
    EQUATIONS_PATTERN = re.compile(
        r'EQUATIONS\s+(\d+)',
        re.IGNORECASE
    )

    @classmethod
    def extract_components(cls, dck_content: str) -> List[Dict[str, Any]]:
        """
        Extract component information from DCK content.

        Returns:
            List of component dictionaries with unit_number, type_number, name.
        """
        components = []
        for match in cls.UNIT_PATTERN.finditer(dck_content):
            components.append({
                "unit_number": int(match.group(1)),
                "type_number": int(match.group(2)),
                "name": match.group(3),
            })
        return components

    @classmethod
    def extract_type_numbers(cls, dck_content: str) -> List[int]:
        """Extract all Type numbers from DCK content."""
        components = cls.extract_components(dck_content)
        return [c["type_number"] for c in components]

    @classmethod
    def extract_connections(cls, dck_content: str) -> List[Tuple[int, int]]:
        """
        Extract connection information from DCK content.

        Returns:
            List of (source_unit, source_output) tuples.
        """
        connections = []

        # Find all INPUTS sections
        lines = dck_content.split('\n')
        in_inputs_section = False
        inputs_count = 0
        current_count = 0

        for line in lines:
            line = line.strip()

            # Check for INPUTS declaration
            inputs_match = cls.INPUTS_PATTERN.match(line)
            if inputs_match:
                in_inputs_section = True
                inputs_count = int(inputs_match.group(1))
                current_count = 0
                continue

            # Skip if not in inputs section
            if not in_inputs_section:
                continue

            # Check for end of inputs section
            if line.startswith('***') or line.startswith('UNIT') or line.startswith('END'):
                in_inputs_section = False
                continue

            # Parse connection lines
            conn_matches = cls.CONNECTION_PATTERN.findall(line)
            for match in conn_matches:
                source_unit = int(match[0])
                source_output = int(match[1])
                if source_unit != 0:  # 0,0 means unconnected
                    connections.append((source_unit, source_output))
                current_count += 1

            if current_count >= inputs_count:
                in_inputs_section = False

        return connections

    @classmethod
    def count_connections(cls, dck_content: str) -> int:
        """Count total number of valid connections (excluding 0,0)."""
        return len(cls.extract_connections(dck_content))

    @classmethod
    def extract_parameters(cls, dck_content: str) -> Dict[int, int]:
        """
        Extract parameter counts per unit.

        Returns:
            Dictionary mapping unit_number to parameter_count.
        """
        params = {}
        lines = dck_content.split('\n')
        current_unit = None

        for i, line in enumerate(lines):
            line = line.strip()

            # Track current unit
            unit_match = cls.UNIT_PATTERN.match(line)
            if unit_match:
                current_unit = int(unit_match.group(1))
                continue

            # Check for PARAMETERS declaration
            params_match = cls.PARAMETERS_PATTERN.match(line)
            if params_match and current_unit is not None:
                params[current_unit] = int(params_match.group(1))

        return params

    @classmethod
    def extract_equations_count(cls, dck_content: str) -> int:
        """Count total number of equations."""
        total = 0
        for match in cls.EQUATIONS_PATTERN.finditer(dck_content):
            total += int(match.group(1))
        return total


class DCKMetricsCalculator:
    """
    Calculator for DCK evaluation metrics.

    Calculates all 5 metrics:
    - SC (Syntax Correctness)
    - CA (Component Accuracy)
    - CC (Connection Completeness)
    - PR (Parameter Reasonableness)
    - ESR (Execution Success Rate)
    """

    def __init__(self, tmf_data_path: Path = None):
        """
        Initialize the metrics calculator.

        Args:
            tmf_data_path: Path to TMF parsed JSON file.
        """
        self._tmf_data_path = tmf_data_path
        self._tmf_specs: Dict[str, Dict[str, Any]] = {}
        self._format_checker = None
        self._validator = None

    @property
    def format_checker(self):
        """Get or create format checker."""
        if self._format_checker is None:
            try:
                from ..dck.format_checker import get_format_checker
                self._format_checker = get_format_checker()
            except Exception as e:
                logger.warning(f"Could not create format checker: {e}")
        return self._format_checker

    @property
    def validator(self):
        """Get or create DCK validator."""
        if self._validator is None:
            try:
                from ..dck.validator import get_validator
                self._validator = get_validator()
            except Exception as e:
                logger.warning(f"Could not create validator: {e}")
        return self._validator

    def _load_tmf_specs(self):
        """Load TMF specifications."""
        if self._tmf_specs:
            return

        if self._tmf_data_path is None:
            try:
                from ..core.config import get_settings
                settings = get_settings()
                self._tmf_data_path = settings.tmf_parsed_path
            except Exception as e:
                logger.warning(f"Could not get TMF data path: {e}")
                return

        if not self._tmf_data_path or not Path(self._tmf_data_path).exists():
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
                return

            # Index by type number
            for tmf in tmf_list:
                if isinstance(tmf, dict):
                    type_num = tmf.get("type_number") or tmf.get("type") or tmf.get("Type")
                    if type_num:
                        type_key = str(type_num).replace("Type", "").strip()
                        self._tmf_specs[type_key] = tmf

            logger.info(f"Loaded TMF specs for {len(self._tmf_specs)} types")

        except Exception as e:
            logger.error(f"Error loading TMF data: {e}")

    def calculate_syntax_correctness(self, dck_content: str) -> MetricResult:
        """
        Calculate Syntax Correctness (SC) metric.

        SC = 1.0 if no format errors, 0.0 otherwise.

        Args:
            dck_content: Generated DCK content.

        Returns:
            MetricResult for SC.
        """
        errors = []
        details = {}

        try:
            if self.format_checker:
                result = self.format_checker.validate_complete_dck(dck_content)
                error_count = len([e for e in result.errors if e.severity.value == "error"])
                warning_count = len(result.warnings)

                details = {
                    "error_count": error_count,
                    "warning_count": warning_count,
                    "errors": [str(e) for e in result.errors[:5]],  # First 5 errors
                }

                # SC is 1.0 if no errors, 0.0 otherwise
                value = 1.0 if error_count == 0 else 0.0
                passed = error_count == 0

            else:
                # Fallback: basic syntax check
                has_version = "VERSION" in dck_content.upper()
                has_end = dck_content.strip().upper().endswith("END") or "\nEND\n" in dck_content.upper()
                has_unit = "UNIT" in dck_content.upper()

                value = 1.0 if (has_version and has_end and has_unit) else 0.0
                passed = value == 1.0
                details = {
                    "has_version": has_version,
                    "has_end": has_end,
                    "has_unit": has_unit,
                }

        except Exception as e:
            logger.error(f"Error calculating SC: {e}")
            value = 0.0
            passed = False
            errors.append(str(e))

        return MetricResult(
            metric_type=MetricType.SC,
            value=value,
            passed=passed,
            details=details,
            errors=errors,
        )

    def calculate_component_accuracy(
        self,
        generated_dck: str,
        reference_dck: str,
    ) -> MetricResult:
        """
        Calculate Component Accuracy (CA) metric.

        CA = |generated_types ∩ expected_types| / |expected_types|

        Args:
            generated_dck: Generated DCK content.
            reference_dck: Reference DCK content.

        Returns:
            MetricResult for CA.
        """
        errors = []
        details = {}

        try:
            # Extract type numbers
            generated_types = set(DCKParser.extract_type_numbers(generated_dck))
            expected_types = set(DCKParser.extract_type_numbers(reference_dck))

            # Calculate accuracy
            if not expected_types:
                value = 1.0  # No expected types, consider as perfect
            else:
                matched = len(generated_types & expected_types)
                value = matched / len(expected_types)

            # Calculate additional metrics
            missing_types = expected_types - generated_types
            extra_types = generated_types - expected_types

            details = {
                "generated_types": sorted(list(generated_types)),
                "expected_types": sorted(list(expected_types)),
                "matched_count": len(generated_types & expected_types),
                "missing_types": sorted(list(missing_types)),
                "extra_types": sorted(list(extra_types)),
            }

            passed = value >= 0.8  # 80% threshold

        except Exception as e:
            logger.error(f"Error calculating CA: {e}")
            value = 0.0
            passed = False
            errors.append(str(e))

        return MetricResult(
            metric_type=MetricType.CA,
            value=value,
            passed=passed,
            details=details,
            errors=errors,
        )

    def calculate_connection_completeness(
        self,
        generated_dck: str,
        reference_dck: str,
    ) -> MetricResult:
        """
        Calculate Connection Completeness (CC) metric.

        CC = min(generated_connections / expected_connections, 1.0)

        Args:
            generated_dck: Generated DCK content.
            reference_dck: Reference DCK content.

        Returns:
            MetricResult for CC.
        """
        errors = []
        details = {}

        try:
            # Count connections
            generated_count = DCKParser.count_connections(generated_dck)
            expected_count = DCKParser.count_connections(reference_dck)

            # Calculate completeness
            if expected_count == 0:
                value = 1.0  # No expected connections
            else:
                value = min(generated_count / expected_count, 1.0)

            details = {
                "generated_connections": generated_count,
                "expected_connections": expected_count,
                "ratio": value,
            }

            passed = value >= 0.7  # 70% threshold

        except Exception as e:
            logger.error(f"Error calculating CC: {e}")
            value = 0.0
            passed = False
            errors.append(str(e))

        return MetricResult(
            metric_type=MetricType.CC,
            value=value,
            passed=passed,
            details=details,
            errors=errors,
        )

    def calculate_parameter_reasonableness(
        self,
        dck_content: str,
    ) -> MetricResult:
        """
        Calculate Parameter Reasonableness (PR) metric.

        PR = components_with_correct_param_count / total_components

        Args:
            dck_content: Generated DCK content.

        Returns:
            MetricResult for PR.
        """
        errors = []
        details = {}

        try:
            self._load_tmf_specs()

            # Extract components and their parameters
            components = DCKParser.extract_components(dck_content)
            params = DCKParser.extract_parameters(dck_content)

            if not components:
                return MetricResult(
                    metric_type=MetricType.PR,
                    value=1.0,
                    passed=True,
                    details={"message": "No components to validate"},
                    errors=[],
                )

            # Check each component's parameter count against TMF
            correct_count = 0
            issues = []

            for comp in components:
                unit_num = comp["unit_number"]
                type_num = str(comp["type_number"])
                actual_params = params.get(unit_num, 0)

                # Get expected parameter count from TMF
                tmf_spec = self._tmf_specs.get(type_num, {})
                expected_params = None

                # Try different field names for parameter count
                for field_name in ["parameter_count", "parameters", "num_parameters", "n_parameters"]:
                    if field_name in tmf_spec:
                        expected_params = tmf_spec[field_name]
                        break

                if expected_params is not None:
                    if actual_params == expected_params:
                        correct_count += 1
                    else:
                        issues.append(
                            f"UNIT {unit_num} (Type{type_num}): "
                            f"expected {expected_params} params, got {actual_params}"
                        )
                else:
                    # No TMF spec available, assume correct
                    correct_count += 1

            value = correct_count / len(components) if components else 1.0

            details = {
                "total_components": len(components),
                "correct_param_count": correct_count,
                "issues": issues[:5],  # First 5 issues
            }

            passed = value >= 0.8  # 80% threshold

        except Exception as e:
            logger.error(f"Error calculating PR: {e}")
            value = 0.0
            passed = False
            errors.append(str(e))

        return MetricResult(
            metric_type=MetricType.PR,
            value=value,
            passed=passed,
            details=details,
            errors=errors,
        )

    def calculate_execution_success(
        self,
        dck_content: str,
        task_id: str = "",
    ) -> MetricResult:
        """
        Calculate Execution Success Rate (ESR) metric.

        ESR = 1.0 if TrnEXE64.exe runs successfully, 0.0 otherwise.

        This is the MOST IMPORTANT metric.
        Proposed method must achieve ESR=100% on all 30 test cases.

        Args:
            dck_content: Generated DCK content.
            task_id: Task identifier for logging.

        Returns:
            MetricResult for ESR.
        """
        errors = []
        details = {}

        try:
            if self.validator:
                result = self.validator.validate(
                    dck_content=dck_content,
                    task_id=task_id,
                    skip_format_check=False,
                    analyze_errors=True,
                )

                value = 1.0 if result.success else 0.0
                passed = result.success

                details = {
                    "format_valid": result.format_valid,
                    "execution_valid": result.execution_valid,
                    "error_count": result.error_count,
                    "execution_time": result.execution_time,
                }

                if not result.success:
                    # Add error details
                    errors.extend([str(e) for e in result.trnsys_errors[:5]])
                    if result.error_analysis:
                        details["root_cause"] = result.error_analysis.root_cause
                        details["fix_suggestions"] = result.error_analysis.fix_suggestions

            else:
                # Cannot validate without validator
                value = 0.0
                passed = False
                errors.append("DCK validator not available")

        except Exception as e:
            logger.error(f"Error calculating ESR: {e}")
            value = 0.0
            passed = False
            errors.append(str(e))

        return MetricResult(
            metric_type=MetricType.ESR,
            value=value,
            passed=passed,
            details=details,
            errors=errors,
        )

    def calculate_all_metrics(
        self,
        generated_dck: str,
        reference_dck: str,
        task_id: str = "",
    ) -> EvaluationResult:
        """
        Calculate all 5 metrics for a single test case.

        Args:
            generated_dck: Generated DCK content.
            reference_dck: Reference DCK content.
            task_id: Task identifier.

        Returns:
            Complete EvaluationResult.
        """
        # Calculate individual metrics
        sc = self.calculate_syntax_correctness(generated_dck)
        ca = self.calculate_component_accuracy(generated_dck, reference_dck)
        cc = self.calculate_connection_completeness(generated_dck, reference_dck)
        pr = self.calculate_parameter_reasonableness(generated_dck)
        esr = self.calculate_execution_success(generated_dck, task_id)

        # Overall success is determined by ESR
        success = esr.passed

        return EvaluationResult(
            task_id=task_id,
            success=success,
            syntax_correctness=sc,
            component_accuracy=ca,
            connection_completeness=cc,
            parameter_reasonableness=pr,
            execution_success=esr,
            generated_dck=generated_dck,
            reference_dck=reference_dck,
        )

    def aggregate_results(
        self,
        results: List[EvaluationResult],
    ) -> AggregatedMetrics:
        """
        Aggregate metrics across multiple test cases.

        Args:
            results: List of individual evaluation results.

        Returns:
            AggregatedMetrics with averages and ESR.
        """
        if not results:
            return AggregatedMetrics(
                total_cases=0,
                passed_cases=0,
                results=[],
            )

        total = len(results)
        passed = sum(1 for r in results if r.success)

        # Calculate averages
        avg_sc = sum(r.syntax_correctness.value for r in results if r.syntax_correctness) / total
        avg_ca = sum(r.component_accuracy.value for r in results if r.component_accuracy) / total
        avg_cc = sum(r.connection_completeness.value for r in results if r.connection_completeness) / total
        avg_pr = sum(r.parameter_reasonableness.value for r in results if r.parameter_reasonableness) / total

        # ESR is passed_cases / total_cases
        esr = passed / total

        return AggregatedMetrics(
            total_cases=total,
            passed_cases=passed,
            avg_syntax_correctness=avg_sc,
            avg_component_accuracy=avg_ca,
            avg_connection_completeness=avg_cc,
            avg_parameter_reasonableness=avg_pr,
            execution_success_rate=esr,
            results=results,
        )


# === Convenience Functions ===

def get_metrics_calculator(tmf_data_path: Path = None) -> DCKMetricsCalculator:
    """Get a metrics calculator instance."""
    return DCKMetricsCalculator(tmf_data_path=tmf_data_path)


def calculate_esr(results: List[EvaluationResult]) -> float:
    """
    Calculate Execution Success Rate from results.

    ESR = passed_cases / total_cases

    Args:
        results: List of evaluation results.

    Returns:
        ESR value (0.0 to 1.0).
    """
    if not results:
        return 0.0
    passed = sum(1 for r in results if r.success)
    return passed / len(results)


def check_esr_100_percent(results: List[EvaluationResult]) -> bool:
    """
    Check if ESR is 100%.

    This is the hard requirement for Proposed method.

    Args:
        results: List of evaluation results.

    Returns:
        True if all cases passed (ESR=100%).
    """
    if not results:
        return False
    return all(r.success for r in results)


def generate_metrics_report(
    aggregated: AggregatedMetrics,
    method_name: str = "Proposed",
) -> str:
    """
    Generate a human-readable metrics report.

    Args:
        aggregated: Aggregated metrics.
        method_name: Name of the method.

    Returns:
        Formatted report string.
    """
    lines = [
        "=" * 60,
        f"Evaluation Report: {method_name}",
        "=" * 60,
        "",
        f"Total Test Cases: {aggregated.total_cases}",
        f"Passed Cases: {aggregated.passed_cases}",
        "",
        "Metrics:",
        f"  SC (Syntax Correctness):      {aggregated.avg_syntax_correctness:.2%}",
        f"  CA (Component Accuracy):      {aggregated.avg_component_accuracy:.2%}",
        f"  CC (Connection Completeness): {aggregated.avg_connection_completeness:.2%}",
        f"  PR (Parameter Reasonableness):{aggregated.avg_parameter_reasonableness:.2%}",
        f"  ESR (Execution Success Rate): {aggregated.execution_success_rate:.2%}",
        "",
    ]

    if aggregated.esr_100_percent:
        lines.append("✅ ESR=100% - HARD REQUIREMENT MET!")
    else:
        lines.append(f"❌ ESR={aggregated.execution_success_rate:.2%} - HARD REQUIREMENT NOT MET")
        lines.append(f"   Need {aggregated.total_cases - aggregated.passed_cases} more cases to pass")

    lines.extend([
        "",
        "=" * 60,
    ])

    return "\n".join(lines)
