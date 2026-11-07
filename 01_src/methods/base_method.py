"""
Base Method Class for TRNSYS DCK Generation.

Defines the abstract interface that all four experimental methods must implement:
- B1: DirectLLM (single LLM call)
- B2: RAGOnly (LLM + RAG)
- B3: MultiAgent_NoMemory (multi-agent without memory)
- Proposed: Full framework (multi-agent + RAG + Memory)

Each method must implement the generate() method which takes a task description
and produces a DCK file.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from ..core.config import Settings, get_settings
from ..core.trnsys_runtime import build_trnsys_cmd


@dataclass
class GenerationResult:
    """Result of a DCK generation attempt."""

    # === Core Output ===
    dck_content: str = ""  # Generated DCK file content
    success: bool = False  # Whether generation was successful

    # === Validation Results ===
    validation_passed: bool = False  # Whether TrnEXE64.exe ran successfully
    validation_errors: list[str] = field(default_factory=list)

    # === Metrics ===
    # SC: Syntax Correctness
    syntax_correct: bool = False
    syntax_errors: list[str] = field(default_factory=list)

    # CA: Component Accuracy
    component_accuracy: float = 0.0
    expected_components: list[str] = field(default_factory=list)
    generated_components: list[str] = field(default_factory=list)

    # CC: Connection Completeness
    connection_completeness: float = 0.0
    expected_connections: int = 0
    generated_connections: int = 0

    # PR: Parameter Reasonableness
    parameter_reasonableness: float = 0.0
    parameter_issues: list[str] = field(default_factory=list)

    # ESR: Execution Success Rate (most important)
    execution_success: bool = False
    execution_log: str = ""

    # === Timing ===
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0

    # === Token Usage ===
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    # === Intermediate Results ===
    components_selected: list[dict[str, Any]] = field(default_factory=list)
    connections_generated: list[dict[str, Any]] = field(default_factory=list)
    equations_generated: list[dict[str, Any]] = field(default_factory=list)

    # === Debug Info ===
    iterations: int = 0
    error_history: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "dck_content": self.dck_content,
            "success": self.success,
            "validation_passed": self.validation_passed,
            "validation_errors": self.validation_errors,
            "metrics": {
                "syntax_correct": self.syntax_correct,
                "component_accuracy": self.component_accuracy,
                "connection_completeness": self.connection_completeness,
                "parameter_reasonableness": self.parameter_reasonableness,
                "execution_success": self.execution_success,
            },
            "timing": {
                "duration_seconds": self.duration_seconds,
            },
            "token_usage": {
                "total": self.total_tokens,
                "prompt": self.prompt_tokens,
                "completion": self.completion_tokens,
            },
            "iterations": self.iterations,
        }


class BaseMethod(ABC):
    """
    Abstract base class for DCK generation methods.

    All four experimental methods (B1, B2, B3, Proposed) inherit from this class
    and implement the generate() method according to their specific approach.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        name: str = "BaseMethod",
    ):
        """
        Initialize the method.

        Args:
            settings: Configuration settings
            name: Human-readable name for the method
        """
        self.settings = settings or get_settings()
        self.name = name

    @abstractmethod
    async def generate(
        self,
        task_description: str,
        task_id: Optional[str] = None,
    ) -> GenerationResult:
        """
        Generate a DCK file from a task description.

        This is the main entry point for DCK generation. Each method
        implements this differently:
        - B1: Single LLM call
        - B2: RAG retrieval + LLM call
        - B3: Multi-agent collaboration (no memory)
        - Proposed: Multi-agent + RAG + Memory

        Args:
            task_description: Natural language description of the system to model
            task_id: Optional unique identifier for tracking

        Returns:
            GenerationResult containing the generated DCK and metrics
        """
        pass

    @abstractmethod
    def get_method_info(self) -> dict[str, Any]:
        """
        Get information about this method.

        Returns:
            Dictionary with method configuration and capabilities
        """
        pass

    async def validate_dck(
        self,
        dck_content: str,
        output_dir: Optional[Path] = None,
    ) -> tuple[bool, list[str], str]:
        """
        Validate a DCK file by running TrnEXE64.exe.

        Args:
            dck_content: DCK file content
            output_dir: Directory to write temporary files

        Returns:
            Tuple of (success, error_list, log_content)
        """
        import subprocess
        import tempfile

        if output_dir is None:
            output_dir = Path(tempfile.mkdtemp())

        # Write DCK file
        dck_path = output_dir / "test.dck"
        dck_path.write_text(dck_content, encoding="utf-8")

        # Run TrnEXE64.exe
        try:
            # Use the centrally configured timeout (single source of truth).
            from ..core.performance_config import get_performance_config
            exec_timeout = get_performance_config().trnsys_validation_timeout

            result = subprocess.run(
                build_trnsys_cmd(self.settings.trnsys_exe_path, str(dck_path)),
                capture_output=True,
                text=True,
                timeout=exec_timeout,  # configurable via performance_config
                cwd=str(output_dir),
            )

            # Check for log and lst files
            log_path = output_dir / "test.log"
            lst_path = output_dir / "test.lst"

            log_content = ""
            errors = []

            if log_path.exists():
                log_content = log_path.read_text(encoding="utf-8", errors="ignore")
                # Parse errors from log
                for line in log_content.split("\n"):
                    if "error" in line.lower() or "fatal" in line.lower():
                        errors.append(line.strip())

            if lst_path.exists():
                lst_content = lst_path.read_text(encoding="utf-8", errors="ignore")
                # Parse errors from lst
                for line in lst_content.split("\n"):
                    if "***" in line and "error" in line.lower():
                        errors.append(line.strip())

            success = len(errors) == 0 and result.returncode == 0

            return success, errors, log_content

        except subprocess.TimeoutExpired:
            return False, ["Simulation timed out after 5 minutes"], ""
        except FileNotFoundError:
            return False, [f"TrnEXE64.exe not found at {self.settings.trnsys_exe_path}"], ""
        except Exception as e:
            return False, [f"Validation error: {str(e)}"], ""

    def _create_task_id(self) -> str:
        """Create a unique task ID."""
        import uuid
        return f"{self.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


class MethodRegistry:
    """Registry for managing available methods."""

    _methods: dict[str, type[BaseMethod]] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a method class."""
        def decorator(method_class: type[BaseMethod]):
            cls._methods[name] = method_class
            return method_class
        return decorator

    @classmethod
    def get(cls, name: str) -> Optional[type[BaseMethod]]:
        """Get a method class by name."""
        return cls._methods.get(name)

    @classmethod
    def list_methods(cls) -> list[str]:
        """List all registered method names."""
        return list(cls._methods.keys())

    @classmethod
    def create(cls, name: str, **kwargs) -> Optional[BaseMethod]:
        """Create an instance of a method by name."""
        method_class = cls.get(name)
        if method_class:
            return method_class(**kwargs)
        return None
