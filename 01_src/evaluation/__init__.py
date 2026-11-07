"""
Evaluation module - Metrics and experiment runner.

This module provides:
1. 5 evaluation metrics (SC, CA, CC, PR, ESR)
2. Experiment runner for pilot testing and full evaluation
3. Result aggregation and reporting

Key Classes:
- DCKMetricsCalculator: Calculates all 5 metrics
- EvaluationRunner: Runs experiments and saves results
- AggregatedMetrics: Aggregates results across test cases

Key Functions:
- calculate_esr(): Calculate Execution Success Rate
- check_esr_100_percent(): Verify ESR=100% (hard requirement)
- run_pilot_test(): Run pilot testing
- run_full_evaluation(): Run full 30-case evaluation
"""

from .metrics import (
    # Enums
    MetricType,
    # Data classes
    MetricResult,
    EvaluationResult,
    AggregatedMetrics,
    # Calculator
    DCKMetricsCalculator,
    DCKParser,
    # Functions
    get_metrics_calculator,
    calculate_esr,
    check_esr_100_percent,
    generate_metrics_report,
)

from .runner import (
    # Data classes
    TestCase,
    RunResult,
    PilotTestResult,
    # Runner
    EvaluationRunner,
    # Functions
    create_evaluation_runner,
    run_pilot_test,
    run_full_evaluation,
    verify_esr_100_percent,
)

__all__ = [
    # Metrics
    "MetricType",
    "MetricResult",
    "EvaluationResult",
    "AggregatedMetrics",
    "DCKMetricsCalculator",
    "DCKParser",
    "get_metrics_calculator",
    "calculate_esr",
    "check_esr_100_percent",
    "generate_metrics_report",
    # Runner
    "TestCase",
    "RunResult",
    "PilotTestResult",
    "EvaluationRunner",
    "create_evaluation_runner",
    "run_pilot_test",
    "run_full_evaluation",
    "verify_esr_100_percent",
]
