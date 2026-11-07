"""
Evaluation Runner for TRNSYS DCK Generator.

This module provides the experiment runner for evaluating DCK generation methods.
Supports:
1. Single test case execution
2. Pilot testing (1/5/10/15/30 cases)
3. Full evaluation (30 cases)
4. Result saving and report generation

Key Features:
- Progressive testing (1→5→10→15→30)
- ESR=100% verification for Proposed method
- Detailed result logging
- Comparison with reference DCK files
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from ..core.config import Settings, get_settings
from ..methods.base_method import BaseMethod, GenerationResult, MethodRegistry
from .metrics import (
    AggregatedMetrics,
    DCKMetricsCalculator,
    EvaluationResult,
    MetricResult,
    check_esr_100_percent,
    generate_metrics_report,
    get_metrics_calculator,
)

logger = logging.getLogger(__name__)


@dataclass
class TestCase:
    """Represents a single test case."""

    task_id: str
    task_description: str
    reference_dck_path: Optional[Path] = None
    reference_dck_content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def load_reference_dck(self) -> str:
        """Load reference DCK content from file."""
        if self.reference_dck_content:
            return self.reference_dck_content

        if self.reference_dck_path and self.reference_dck_path.exists():
            try:
                self.reference_dck_content = self.reference_dck_path.read_text(
                    encoding="utf-8", errors="ignore"
                )
            except Exception as e:
                logger.warning(f"Could not load reference DCK: {e}")

        return self.reference_dck_content


@dataclass
class RunResult:
    """Result of a single test run."""

    task_id: str
    success: bool
    generation_result: Optional[GenerationResult] = None
    evaluation_result: Optional[EvaluationResult] = None
    error: Optional[str] = None
    duration: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_id": self.task_id,
            "success": self.success,
            "error": self.error,
            "duration": self.duration,
            "generation": self.generation_result.to_dict() if self.generation_result else None,
            "evaluation": self.evaluation_result.to_dict() if self.evaluation_result else None,
        }


@dataclass
class PilotTestResult:
    """Result of pilot testing."""

    num_cases: int
    passed_cases: int
    esr: float
    esr_100_percent: bool
    results: List[RunResult] = field(default_factory=list)
    failed_cases: List[str] = field(default_factory=list)
    duration: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "num_cases": self.num_cases,
            "passed_cases": self.passed_cases,
            "esr": self.esr,
            "esr_100_percent": self.esr_100_percent,
            "failed_cases": self.failed_cases,
            "duration": self.duration,
        }


class EvaluationRunner:
    """
    Runner for evaluating DCK generation methods.

    Supports:
    - Single test execution
    - Pilot testing (progressive: 1→5→10→15→30)
    - Full evaluation (30 cases)
    - Result saving and reporting
    """

    # Pilot test case counts
    PILOT_STAGES = [1, 5, 10, 15, 30]

    def __init__(
        self,
        method: str = "proposed",
        settings: Optional[Settings] = None,
        output_dir: Optional[Path] = None,
    ):
        """
        Initialize the evaluation runner.

        Args:
            method: Method name ("proposed", "b1_direct_llm", "b2_rag", "b3_multiagent_rag").
            settings: Configuration settings.
            output_dir: Directory for saving results.
        """
        self.method_name = method
        self.settings = settings or get_settings()
        self.output_dir = output_dir or self._get_default_output_dir()

        # Lazy initialization
        self._method_instance = None
        self._metrics_calculator = None
        self._test_cases: List[TestCase] = []

    def _get_default_output_dir(self) -> Path:
        """Get default output directory."""
        base_dir = Path(self.settings.project_root) / "data" / "results"
        return base_dir / self.method_name

    @property
    def method_instance(self) -> BaseMethod:
        """Get or create method instance."""
        if self._method_instance is None:
            self._method_instance = self._create_method_instance()
        return self._method_instance

    def _create_method_instance(self) -> BaseMethod:
        """Create the method instance based on method name."""
        if self.method_name == "proposed":
            from ..methods.proposed_full import create_proposed_method
            return create_proposed_method(settings=self.settings)

        # Try registry
        method_class = MethodRegistry.get(self.method_name)
        if method_class:
            return method_class(settings=self.settings)

        raise ValueError(f"Unknown method: {self.method_name}")

    @property
    def metrics_calculator(self) -> DCKMetricsCalculator:
        """Get or create metrics calculator."""
        if self._metrics_calculator is None:
            self._metrics_calculator = get_metrics_calculator()
        return self._metrics_calculator

    def load_test_cases(self, num_cases: int = 30) -> List[TestCase]:
        """
        Load test cases from the test task descriptions.

        Args:
            num_cases: Number of cases to load (max 30).

        Returns:
            List of TestCase objects.
        """
        if self._test_cases and len(self._test_cases) >= num_cases:
            return self._test_cases[:num_cases]

        test_cases = []

        # Load from test task descriptions
        task_desc_dir = Path(self.settings.project_root) / "data" / "test_task_descriptions"

        if not task_desc_dir.exists():
            logger.warning(f"Task descriptions directory not found: {task_desc_dir}")
            return []

        # Load all task description files
        for i in range(1, min(num_cases + 1, 31)):
            task_id = f"test_{i:03d}"

            # Find task description file
            desc_files = list(task_desc_dir.glob(f"{task_id}_*.json"))
            if not desc_files:
                logger.warning(f"No task description found for {task_id}")
                continue

            desc_file = desc_files[0]

            try:
                with open(desc_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                task_description = data.get("task_description", "")
                if not task_description:
                    task_description = data.get("description", "")

                # Find reference DCK file
                ref_dck_path = self._find_reference_dck(task_id, data)

                test_case = TestCase(
                    task_id=task_id,
                    task_description=task_description,
                    reference_dck_path=ref_dck_path,
                    metadata=data,
                )

                test_cases.append(test_case)

            except Exception as e:
                logger.error(f"Error loading test case {task_id}: {e}")

        self._test_cases = test_cases
        logger.info(f"Loaded {len(test_cases)} test cases")

        return test_cases[:num_cases]

    def _find_reference_dck(
        self,
        task_id: str,
        metadata: Dict[str, Any],
    ) -> Optional[Path]:
        """Find the reference DCK file for a test case."""
        # Try to get from metadata
        dck_filename = metadata.get("dck_filename", "")
        if dck_filename:
            # Look in test set directory
            test_dck_dir = Path(self.settings.original_data_root) / "206_dck_files" / "test_set"
            ref_path = test_dck_dir / dck_filename
            if ref_path.exists():
                return ref_path

        # Try to find by task_id pattern
        test_dck_dir = Path(self.settings.original_data_root) / "206_dck_files" / "test_set"
        if test_dck_dir.exists():
            # List all DCK files and try to match
            dck_files = list(test_dck_dir.glob("*.dck"))
            # Return first match or None
            if dck_files:
                # Try to match by index
                idx = int(task_id.split("_")[1])
                if idx <= len(dck_files):
                    return dck_files[idx - 1]

        return None

    async def run_single_test(
        self,
        test_case: TestCase,
        save_result: bool = True,
    ) -> RunResult:
        """
        Run a single test case.

        Args:
            test_case: The test case to run.
            save_result: Whether to save the result.

        Returns:
            RunResult with generation and evaluation results.
        """
        start_time = time.time()
        task_id = test_case.task_id

        logger.info(f"Running test case: {task_id}")

        try:
            # Generate DCK
            generation_result = await self.method_instance.generate(
                task_description=test_case.task_description,
                task_id=task_id,
            )

            # Load reference DCK for comparison
            reference_dck = test_case.load_reference_dck()

            # Calculate metrics
            if generation_result.dck_content:
                evaluation_result = self.metrics_calculator.calculate_all_metrics(
                    generated_dck=generation_result.dck_content,
                    reference_dck=reference_dck,
                    task_id=task_id,
                )
            else:
                evaluation_result = EvaluationResult(
                    task_id=task_id,
                    success=False,
                )

            duration = time.time() - start_time

            run_result = RunResult(
                task_id=task_id,
                success=generation_result.success,
                generation_result=generation_result,
                evaluation_result=evaluation_result,
                duration=duration,
            )

            # Save result
            if save_result:
                self._save_single_result(run_result, generation_result.dck_content)

            return run_result

        except Exception as e:
            logger.error(f"Test case {task_id} failed: {e}")
            return RunResult(
                task_id=task_id,
                success=False,
                error=str(e),
                duration=time.time() - start_time,
            )

    async def run_pilot_test(
        self,
        num_cases: int = 1,
        save_results: bool = True,
    ) -> PilotTestResult:
        """
        Run pilot testing with specified number of cases.

        Progressive testing: 1→5→10→15→30
        If any case fails, stop and report.

        Args:
            num_cases: Number of cases to test.
            save_results: Whether to save results.

        Returns:
            PilotTestResult with ESR and details.
        """
        start_time = time.time()

        logger.info(f"Starting pilot test with {num_cases} cases")

        # Load test cases
        test_cases = self.load_test_cases(num_cases)

        if len(test_cases) < num_cases:
            logger.warning(
                f"Only {len(test_cases)} test cases available, "
                f"requested {num_cases}"
            )

        results = []
        failed_cases = []

        for test_case in test_cases:
            result = await self.run_single_test(test_case, save_result=save_results)
            results.append(result)

            if not result.success:
                failed_cases.append(result.task_id)
                logger.warning(f"Test case {result.task_id} FAILED")

        # Calculate ESR
        passed = len(results) - len(failed_cases)
        esr = passed / len(results) if results else 0.0
        esr_100 = len(failed_cases) == 0

        duration = time.time() - start_time

        pilot_result = PilotTestResult(
            num_cases=len(results),
            passed_cases=passed,
            esr=esr,
            esr_100_percent=esr_100,
            results=results,
            failed_cases=failed_cases,
            duration=duration,
        )

        # Log summary
        logger.info(
            f"Pilot test completed: {passed}/{len(results)} passed, "
            f"ESR={esr:.2%}, duration={duration:.2f}s"
        )

        if esr_100:
            logger.info("✅ ESR=100% - All cases passed!")
        else:
            logger.warning(
                f"❌ ESR={esr:.2%} - {len(failed_cases)} cases failed: "
                f"{failed_cases}"
            )

        # Save pilot test summary
        if save_results:
            self._save_pilot_summary(pilot_result)

        return pilot_result

    async def run_full_evaluation(
        self,
        save_results: bool = True,
    ) -> AggregatedMetrics:
        """
        Run full evaluation on all 30 test cases.

        This is the final verification for ESR=100% requirement.

        Args:
            save_results: Whether to save results.

        Returns:
            AggregatedMetrics with all metrics.
        """
        logger.info("Starting full evaluation (30 cases)")

        # Run pilot test with all 30 cases
        pilot_result = await self.run_pilot_test(
            num_cases=30,
            save_results=save_results,
        )

        # Convert to evaluation results
        eval_results = []
        for run_result in pilot_result.results:
            if run_result.evaluation_result:
                eval_results.append(run_result.evaluation_result)

        # Aggregate metrics
        aggregated = self.metrics_calculator.aggregate_results(eval_results)

        # Generate and save report
        if save_results:
            report = generate_metrics_report(aggregated, self.method_name)
            self._save_report(report, aggregated)

        # Log final result
        logger.info("\n" + generate_metrics_report(aggregated, self.method_name))

        return aggregated

    def run_progressive_pilot_test(
        self,
        stop_on_failure: bool = True,
    ) -> Dict[int, PilotTestResult]:
        """
        Run progressive pilot testing: 1→5→10→15→30.

        Stops at each stage if ESR<100% (when stop_on_failure=True).

        Args:
            stop_on_failure: Whether to stop if ESR<100%.

        Returns:
            Dictionary mapping stage to PilotTestResult.
        """
        results = {}

        for stage in self.PILOT_STAGES:
            logger.info(f"\n{'='*60}")
            logger.info(f"Progressive Pilot Test: Stage {stage} cases")
            logger.info(f"{'='*60}")

            result = asyncio.run(self.run_pilot_test(num_cases=stage))
            results[stage] = result

            if not result.esr_100_percent:
                logger.warning(
                    f"Stage {stage} failed: ESR={result.esr:.2%}"
                )
                if stop_on_failure:
                    logger.info("Stopping progressive test due to failure")
                    break
            else:
                logger.info(f"Stage {stage} passed: ESR=100%")

        return results

    def _save_single_result(
        self,
        result: RunResult,
        dck_content: str,
    ) -> None:
        """Save a single test result."""
        # Create output directory
        task_dir = self.output_dir / result.task_id
        task_dir.mkdir(parents=True, exist_ok=True)

        # Save generated DCK
        if dck_content:
            dck_path = task_dir / f"{result.task_id}_generated.dck"
            dck_path.write_text(dck_content, encoding="utf-8")

        # Save result JSON
        result_path = task_dir / f"{result.task_id}_result.json"
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False, default=str)

    def _save_pilot_summary(self, result: PilotTestResult) -> None:
        """Save pilot test summary."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path = self.output_dir / f"pilot_test_{result.num_cases}_{timestamp}.json"

        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

    def _save_report(
        self,
        report: str,
        aggregated: AggregatedMetrics,
    ) -> None:
        """Save evaluation report."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save text report
        report_path = self.output_dir / f"evaluation_report_{timestamp}.txt"
        report_path.write_text(report, encoding="utf-8")

        # Save JSON metrics
        metrics_path = self.output_dir / f"evaluation_metrics_{timestamp}.json"
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(aggregated.to_dict(), f, indent=2, ensure_ascii=False)

        logger.info(f"Report saved to: {report_path}")


# === Convenience Functions ===

def create_evaluation_runner(
    method: str = "proposed",
    settings: Optional[Settings] = None,
) -> EvaluationRunner:
    """
    Create an evaluation runner.

    Args:
        method: Method name.
        settings: Configuration settings.

    Returns:
        EvaluationRunner instance.
    """
    return EvaluationRunner(method=method, settings=settings)


async def run_pilot_test(
    method: str = "proposed",
    num_cases: int = 1,
) -> PilotTestResult:
    """
    Run pilot test for a method.

    Args:
        method: Method name.
        num_cases: Number of cases.

    Returns:
        PilotTestResult.
    """
    runner = create_evaluation_runner(method=method)
    return await runner.run_pilot_test(num_cases=num_cases)


async def run_full_evaluation(
    method: str = "proposed",
) -> AggregatedMetrics:
    """
    Run full evaluation for a method.

    Args:
        method: Method name.

    Returns:
        AggregatedMetrics.
    """
    runner = create_evaluation_runner(method=method)
    return await runner.run_full_evaluation()


def verify_esr_100_percent(
    method: str = "proposed",
) -> bool:
    """
    Verify that a method achieves ESR=100%.

    This is the hard requirement for Proposed method.

    Args:
        method: Method name.

    Returns:
        True if ESR=100%.
    """
    runner = create_evaluation_runner(method=method)
    result = asyncio.run(runner.run_full_evaluation())
    return result.esr_100_percent
