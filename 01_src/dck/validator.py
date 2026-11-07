"""
DCK Validator for TRNSYS DCK Generator.

This module provides DCK validation using TrnEXE64.exe.
Integrates format checking with actual TRNSYS execution validation.

Key Features:
1. Pre-validation using format checker
2. TrnEXE64.exe execution for actual validation
3. Log and LST file parsing
4. LLM-driven error analysis
5. Fix suggestion generation

Design Principles:
- LLM-driven error analysis (NOT hardcoded rules)
- Comprehensive error extraction from log/lst files
- Integration with format checker for pre-validation
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from ..core.config import get_settings
from ..core.trnsys_runtime import (
    build_trnsys_cmd,
    trnsys_exe_exists,
    is_host_mode,
    execute_trnsys_on_host,
)
from .format_checker import (
    DCKFormatChecker,
    FormatError,
    ValidationResult as FormatValidationResult,
    get_format_checker,
)

logger = logging.getLogger(__name__)


@dataclass
class TRNSYSError:
    """Represents an error from TRNSYS execution."""

    error_type: str
    message: str
    severity: str = "error"  # error, warning, fatal
    line_number: Optional[int] = None
    unit_number: Optional[int] = None
    source: str = "log"  # log, lst, execution

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "error_type": self.error_type,
            "message": self.message,
            "severity": self.severity,
            "line_number": self.line_number,
            "unit_number": self.unit_number,
            "source": self.source,
        }

    def __str__(self) -> str:
        """String representation."""
        parts = [f"[{self.severity.upper()}] {self.error_type}: {self.message}"]
        if self.line_number:
            parts.append(f" (line {self.line_number})")
        if self.unit_number:
            parts.append(f" (UNIT {self.unit_number})")
        return "".join(parts)


@dataclass
class ErrorAnalysis:
    """LLM-driven error analysis result."""

    root_cause: str
    responsible_agents: List[str]
    fix_suggestions: List[str]
    priority: str = "high"  # high, medium, low
    affected_units: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "root_cause": self.root_cause,
            "responsible_agents": self.responsible_agents,
            "fix_suggestions": self.fix_suggestions,
            "priority": self.priority,
            "affected_units": self.affected_units,
        }


@dataclass
class DCKValidationResult:
    """Complete DCK validation result."""

    success: bool
    format_valid: bool
    execution_valid: bool
    format_errors: List[FormatError] = field(default_factory=list)
    trnsys_errors: List[TRNSYSError] = field(default_factory=list)
    error_analysis: Optional[ErrorAnalysis] = None
    log_content: str = ""
    lst_content: str = ""
    execution_time: float = 0.0
    raw_output: str = ""

    # Persistent file paths (modification 3+4)
    version_tag: str = ""  # e.g. "V1_20260224_060900"
    dck_file_path: str = ""  # Absolute path to saved .dck file
    log_file_path: str = ""  # Absolute path to saved .log file
    lst_file_path: str = ""  # Absolute path to saved .lst file

    @property
    def all_errors(self) -> List[Dict[str, Any]]:
        """Get all errors as dictionaries."""
        errors = []
        for fe in self.format_errors:
            errors.append(fe.to_dict())
        for te in self.trnsys_errors:
            errors.append(te.to_dict())
        return errors

    @property
    def error_count(self) -> int:
        """Total number of errors."""
        return len(self.format_errors) + len(self.trnsys_errors)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "format_valid": self.format_valid,
            "execution_valid": self.execution_valid,
            "format_errors": [e.to_dict() for e in self.format_errors],
            "trnsys_errors": [e.to_dict() for e in self.trnsys_errors],
            "error_analysis": self.error_analysis.to_dict() if self.error_analysis else None,
            "error_count": self.error_count,
            "execution_time": self.execution_time,
            "version_tag": self.version_tag,
            "dck_file_path": self.dck_file_path,
            "log_file_path": self.log_file_path,
            "lst_file_path": self.lst_file_path,
        }


class DCKValidator:
    """
    DCK Validator using TrnEXE64.exe.

    Validates DCK files through format checking and actual TRNSYS execution.
    """

    # Default TrnEXE64 path
    DEFAULT_TRNSYS_PATH = r"D:\TRNSYS18_04\Exe\TrnEXE64.exe"

    def __init__(
        self,
        trnsys_exe_path: str = None,
        format_checker: DCKFormatChecker = None,
        timeout: int = None,  # If None, uses performance_config
    ):
        """
        Initialize the validator.

        Args:
            trnsys_exe_path: Path to TrnEXE64.exe.
            format_checker: Format checker instance.
            timeout: Execution timeout in seconds. If None, uses performance_config.
        """
        self.trnsys_exe_path = trnsys_exe_path or self._get_trnsys_path()
        self.format_checker = format_checker or get_format_checker()

        # Use performance_config if timeout is not explicitly provided
        if timeout is None:
            from ..core.performance_config import get_performance_config
            perf_config = get_performance_config()
            self.timeout = perf_config.trnsys_validation_timeout
        else:
            self.timeout = timeout

        self._llm = None

    def _get_trnsys_path(self) -> str:
        """Get TRNSYS executable path from config or default."""
        try:
            from ..core.config import get_settings
            settings = get_settings()
            return str(settings.trnsys_exe_path)
        except Exception:
            return self.DEFAULT_TRNSYS_PATH

    @property
    def llm(self):
        """Get or create LLM instance."""
        if self._llm is None:
            try:
                from ..core.llm_client import create_llm_client
                self._llm = create_llm_client()
            except Exception as e:
                logger.warning(f"Could not create LLM: {e}")
        return self._llm

    def validate(
        self,
        dck_content: str,
        task_id: str = "",
        skip_format_check: bool = False,
        analyze_errors: bool = True,
    ) -> DCKValidationResult:
        """
        Validate DCK content using TrnEXE64.exe only.

        Format checker errors are excluded from validation_errors because
        template assembly is accurate enough and format_checker may over-diagnose.
        Only TrnEXE64.exe errors matter for validation success/failure.

        Args:
            dck_content: The DCK file content.
            task_id: Task identifier for file naming.
            skip_format_check: Ignored (kept for API compatibility).
            analyze_errors: Use LLM to analyze errors.

        Returns:
            DCKValidationResult with TrnEXE64 validation results only.
        """
        start_time = time.time()

        # Run TRNSYS validation (the only error source now)
        trnsys_errors = []
        execution_valid = False
        log_content = ""
        lst_content = ""
        raw_output = ""
        version_tag = ""
        dck_file_path = ""
        log_file_path = ""
        lst_file_path = ""

        try:
            execution_result = self._run_trnsys(dck_content, task_id)
            trnsys_errors = execution_result["errors"]
            execution_valid = execution_result["success"]
            log_content = execution_result.get("log_content", "")
            lst_content = execution_result.get("lst_content", "")
            raw_output = execution_result.get("raw_output", "")
            version_tag = execution_result.get("version_tag", "")
            dck_file_path = execution_result.get("dck_file_path", "")
            log_file_path = execution_result.get("log_file_path", "")
            lst_file_path = execution_result.get("lst_file_path", "")
        except Exception as e:
            logger.error(f"TRNSYS execution failed: {e}")
            trnsys_errors.append(TRNSYSError(
                error_type="execution_error",
                message=str(e),
                severity="fatal",
                source="execution",
            ))

        # Analyze errors with LLM
        error_analysis = None
        if trnsys_errors and analyze_errors and self.llm:
            try:
                error_analysis = self._analyze_errors_with_llm(
                    trnsys_errors,
                    dck_content,
                    log_content,
                )
            except Exception as e:
                logger.warning(f"Error analysis failed: {e}")

        execution_time = time.time() - start_time

        return DCKValidationResult(
            success=execution_valid,
            format_valid=True,  # No longer checked
            execution_valid=execution_valid,
            format_errors=[],  # Always empty - format_checker excluded
            trnsys_errors=trnsys_errors,
            error_analysis=error_analysis,
            log_content=log_content,
            lst_content=lst_content,
            execution_time=execution_time,
            raw_output=raw_output,
            version_tag=version_tag,
            dck_file_path=dck_file_path,
            log_file_path=log_file_path,
            lst_file_path=lst_file_path,
        )

    def _pre_validate(self, dck_content: str) -> FormatValidationResult:
        """Pre-validate DCK format."""
        return self.format_checker.validate_complete_dck(dck_content)

    def _get_validation_output_dir(self, task_id: str) -> Tuple[Path, str, int]:
        """
        Get or create the persistent output directory for validation files.

        Directory structure:
            trnsys_dck_generator/validation_runs/{task_id}/V{n}_{timestamp}/

        Args:
            task_id: Task identifier for the run folder.

        Returns:
            Tuple of (version_dir, version_tag, version_number).
        """
        from datetime import datetime

        base_dir = Path(__file__).parent.parent.parent / "validation_runs"
        run_dir = base_dir / (task_id if task_id else "default")
        run_dir.mkdir(parents=True, exist_ok=True)

        # Determine next version number by scanning existing V* folders
        existing_versions = []
        if run_dir.exists():
            for d in run_dir.iterdir():
                if d.is_dir() and d.name.startswith("V"):
                    try:
                        v_num = int(d.name.split("_")[0][1:])
                        existing_versions.append(v_num)
                    except (ValueError, IndexError):
                        pass

        next_version = max(existing_versions, default=0) + 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version_tag = f"V{next_version}_{timestamp}"

        version_dir = run_dir / version_tag
        version_dir.mkdir(parents=True, exist_ok=True)

        return version_dir, version_tag, next_version

    def _wait_for_file(
        self,
        file_path: Path,
        process: Optional[subprocess.Popen] = None,
        interval: float = 20.0
    ) -> str:
        """
        Wait for a file to appear and read its content with retry.

        If process is provided, waits until the process completes before final read.
        Otherwise, retries indefinitely until file appears with content.

        TRNSYS writes .log/.lst files asynchronously during execution.
        The file content is only complete after the process finishes.

        Args:
            file_path: Path to the file to read.
            process: Optional subprocess.Popen object to monitor.
            interval: Seconds to wait between retries.

        Returns:
            File content string, or "" if file never appeared.
        """
        attempt = 0

        while True:
            attempt += 1

            # Check if file exists and has content
            if file_path.exists() and file_path.stat().st_size > 0:
                # If process provided, check if it's still running
                if process is not None:
                    poll_result = process.poll()
                    if poll_result is None:
                        # Process still running, log is still being written
                        logger.debug(
                            f"File {file_path.name} exists but process still running "
                            f"(attempt {attempt}), waiting..."
                        )
                    else:
                        # Process finished, read final log
                        logger.info(
                            f"Process completed (return code: {poll_result}), "
                            f"reading final {file_path.name}"
                        )
                        try:
                            return file_path.read_text(encoding='utf-8', errors='ignore')
                        except (OSError, PermissionError) as e:
                            logger.warning(f"Failed to read {file_path.name}: {e}")
                            return ""
                else:
                    # No process to monitor, just read the file
                    try:
                        return file_path.read_text(encoding='utf-8', errors='ignore')
                    except (OSError, PermissionError) as e:
                        logger.debug(f"Read attempt {attempt} for {file_path.name} failed: {e}")
            else:
                # File doesn't exist or is empty
                if process is not None:
                    poll_result = process.poll()
                    if poll_result is not None:
                        # Process finished but no log file
                        logger.warning(
                            f"Process completed but {file_path.name} not found or empty"
                        )
                        return ""

                logger.debug(
                    f"Waiting for {file_path.name} (attempt {attempt})..."
                )

            # Wait before next retry
            time.sleep(interval)

    def _run_trnsys(self, dck_content: str, task_id: str = "") -> Dict[str, Any]:
        """
        Run TrnEXE64.exe to validate DCK.

        Uses Popen to allow long-running simulations without timeout killing.
        Monitors process completion before reading final log files.

        Files (.dck, .log, .lst) are saved permanently to:
            trnsys_dck_generator/validation_runs/{task_id}/V{n}_{timestamp}/

        Args:
            dck_content: DCK file content.
            task_id: Task identifier.

        Returns:
            Dictionary with validation results including file paths.
        """
        # Check if TrnEXE64 exists
        if not trnsys_exe_exists(self.trnsys_exe_path):
            return {
                "success": False,
                "errors": [TRNSYSError(
                    error_type="exe_not_found",
                    message=f"TrnEXE64.exe not found at {self.trnsys_exe_path}",
                    severity="fatal",
                    source="execution",
                )],
                "log_content": "",
                "lst_content": "",
                "raw_output": "",
                "version_tag": "",
                "dck_file_path": "",
                "log_file_path": "",
                "lst_file_path": "",
            }

        # Create persistent output directory
        version_dir, version_tag, _ = self._get_validation_output_dir(task_id)

        dck_filename = f"{version_tag}.dck"
        dck_path = version_dir / dck_filename
        log_path = dck_path.with_suffix(".log")
        lst_path = dck_path.with_suffix(".lst")

        # Write DCK file (permanent)
        # Ensure Windows line endings (CRLF) for TRNSYS compatibility
        # Docker containers use Unix line endings (LF) by default, but TRNSYS requires CRLF
        # Normalize to LF first, then convert to CRLF to avoid double conversion
        dck_content_windows = dck_content.replace('\r\n', '\n').replace('\n', '\r\n')

        # Write with newline='' to prevent Python from doing additional line ending conversion
        with open(dck_path, 'w', encoding='utf-8', newline='') as f:
            f.write(dck_content_windows)

        # Run TrnEXE64 - support both host mode and container mode
        raw_output = ""
        return_code = -1
        process = None
        # Safety limit for container-mode (Wine/native) execution. Use the
        # configured timeout (self.timeout) so the limit stays consistent with
        # the host-mode timeout and is controlled from performance_config.
        max_wait_time = self.timeout  # configurable via performance_config.trnsys_validation_timeout

        try:
            # Check execution mode
            if is_host_mode():
                # ===== Host mode: Execute on Windows host via HTTP API =====
                logger.info(f"Using Windows host execution mode for: {dck_path}")

                result = execute_trnsys_on_host(str(dck_path), "/h", timeout=self.timeout)

                # Extract results from HTTP response
                return_code = result.get("exit_code", -1)
                raw_output = result.get("stdout", "") + result.get("stderr", "")

                logger.info(
                    f"TRNSYS execution completed on Windows host with return code: {return_code}"
                )

                # Note: Do not return here - continue to unified error parsing flow
                # Log/lst files will be read in the unified post-processing section

            else:
                # ===== Container mode: Execute with Wine or native =====
                cmd = build_trnsys_cmd(self.trnsys_exe_path, str(dck_path), "/h")  # /h for hidden mode

                logger.info(f"Starting TRNSYS execution: {' '.join(cmd)}")

                # Use Popen instead of run to avoid timeout killing
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    cwd=str(version_dir),
                )

                # Monitor process with safety timeout
                start_time = time.time()
                while True:
                    poll_result = process.poll()

                    if poll_result is not None:
                        # Process completed
                        return_code = poll_result
                        elapsed = time.time() - start_time
                        logger.info(
                            f"TRNSYS process completed with return code: {return_code} "
                            f"(elapsed: {elapsed:.1f}s)"
                        )

                        # Capture output
                        try:
                            stdout, stderr = process.communicate(timeout=5)
                            raw_output = stdout + stderr
                        except Exception as e:
                            logger.warning(f"Failed to capture process output: {e}")

                        break

                    # Check safety timeout
                    elapsed = time.time() - start_time
                    if elapsed > max_wait_time:
                        logger.error(
                            f"TRNSYS execution exceeded maximum wait time ({max_wait_time}s), "
                            f"terminating process..."
                        )
                        process.terminate()
                        time.sleep(5)
                        if process.poll() is None:
                            process.kill()

                        # Read whatever log exists
                        log_content = self._wait_for_file(log_path, process=None)
                        lst_content = self._wait_for_file(lst_path, process=None)
                        errors = self._parse_log_file(log_content) if log_content else []
                        if not errors:
                            errors = [TRNSYSError(
                                error_type="timeout",
                                message=f"TRNSYS execution exceeded maximum wait time of {max_wait_time} seconds",
                                severity="fatal",
                                source="execution",
                            )]

                        return {
                            "success": False,
                            "errors": errors,
                            "log_content": log_content,
                            "lst_content": lst_content,
                            "raw_output": raw_output,
                            "return_code": -1,
                            "version_tag": version_tag,
                            "dck_file_path": str(dck_path.resolve()),
                            "log_file_path": str(log_path.resolve()) if log_path.exists() else "",
                            "lst_file_path": str(lst_path.resolve()) if lst_path.exists() else "",
                        }

                    # Wait before next check
                    time.sleep(5)  # Check every 5 seconds

        except Exception as e:
            logger.error(f"TRNSYS execution error: {e}")

            # Try to terminate process if it exists
            if process and process.poll() is None:
                try:
                    process.terminate()
                    time.sleep(2)
                    if process.poll() is None:
                        process.kill()
                except Exception:
                    pass

            # Read log/lst even on unexpected errors
            log_content = self._wait_for_file(log_path, process=None)
            lst_content = self._wait_for_file(lst_path, process=None)
            errors = self._parse_log_file(log_content) if log_content else []
            if not errors:
                errors = [TRNSYSError(
                    error_type="execution_error",
                    message=str(e),
                    severity="fatal",
                    source="execution",
                )]

            return {
                "success": False,
                "errors": errors,
                "log_content": log_content,
                "lst_content": lst_content,
                "raw_output": "",
                "version_tag": version_tag,
                "dck_file_path": str(dck_path.resolve()),
                "log_file_path": str(log_path.resolve()) if log_path.exists() else "",
                "lst_file_path": str(lst_path.resolve()) if lst_path.exists() else "",
            }

        # --- Process completed normally, read final log/lst files ---
        logger.info("Reading final log and lst files after process completion...")
        log_content = self._wait_for_file(log_path, process=process)
        lst_content = self._wait_for_file(lst_path, process=process)

        # Parse errors from .log only (.lst excluded to avoid duplication)
        errors = self._parse_log_file(log_content)

        # Check for success: return code 0 and no errors in log
        success = (return_code == 0 and len(errors) == 0)

        return {
            "success": success,
            "errors": errors,
            "log_content": log_content,
            "lst_content": lst_content,
            "raw_output": raw_output,
            "return_code": return_code,
            "version_tag": version_tag,
            "dck_file_path": str(dck_path.resolve()),
            "log_file_path": str(log_path.resolve()) if log_path.exists() else "",
            "lst_file_path": str(lst_path.resolve()) if lst_path.exists() else "",
        }

    def _parse_log_file(self, log_content: str) -> List[TRNSYSError]:
        """
        Parse TRNSYS .log file for errors only (exclude Notice and Warning).

        Follows the standard TRNSYS .log format:
            *** {Severity} at time   :   {time}
                Generated by Unit     :   {unit}
                Generated by Type     :   {type}
                {Message lines}

        Only "Error" and "Fatal Error" entries are collected.
        Each error gets fields: index, severity, time_h, unit_number, type_number, detail_message.

        Returns:
            List of TRNSYSError objects (errors only, no warnings/notices).
        """
        errors = []

        if not log_content:
            return errors

        # Match each *** block: severity, time, then multi-line body until next *** or end
        block_pattern = re.compile(
            r'\*\*\*\s+(.*?)\s+at\s+time\s*:\s*([\d.]+)\s*\n'
            r'(.*?)'
            r'(?=\n\s*\*\*\*|\Z)',
            re.DOTALL
        )

        error_index = 0

        for match in block_pattern.finditer(log_content):
            severity_raw = match.group(1).strip()
            time_h = match.group(2).strip()
            body = match.group(3)

            # Only keep Error and Fatal Error (skip Notice and Warning)
            severity_lower = severity_raw.lower()
            if "error" not in severity_lower:
                continue

            # Normalize severity
            if "fatal" in severity_lower:
                severity = "Fatal Error"
            else:
                severity = "Error"

            # Extract unit number (keep raw string)
            unit_match = re.search(
                r'Generated by Unit\s*:\s*(.+)',
                body, re.IGNORECASE
            )
            unit_number = None
            raw_unit = None
            if unit_match:
                raw_unit = unit_match.group(1).strip()
                if raw_unit.isdigit():
                    unit_number = int(raw_unit)

            # Extract type number (keep raw string)
            type_match = re.search(
                r'Generated by Type\s*:\s*(.+)',
                body, re.IGNORECASE
            )
            type_number = None
            raw_type = None
            if type_match:
                raw_type = type_match.group(1).strip()
                if raw_type.isdigit():
                    type_number = int(raw_type)

            # Extract detail_message and reported_information separately
            detail_message_line = ""
            reported_info = ""
            for line in body.split('\n'):
                stripped = line.strip()
                if not stripped:
                    continue
                if stripped.startswith('Generated by Unit') or stripped.startswith('Generated by Type'):
                    continue
                # Keep "TRNSYS Message" / "Message" lines as-is for detail_message
                if stripped.startswith('TRNSYS Message') or stripped.startswith('Message'):
                    detail_message_line = stripped
                elif stripped.startswith('Reported information'):
                    # Extract value after "Reported information  : "
                    colon_idx = stripped.find(':')
                    if colon_idx >= 0:
                        reported_info = stripped[colon_idx + 1:].strip()

            # Fallback: if no TRNSYS Message line, use severity as detail
            if not detail_message_line:
                detail_message_line = severity_raw

            error_index += 1

            errors.append(TRNSYSError(
                error_type=severity,
                message=detail_message_line[:500],
                severity=severity.lower().replace(" ", "_"),
                unit_number=unit_number,
                source="log",
            ))
            # Store extra fields as attributes for the new format
            errors[-1]._index = error_index
            errors[-1]._time_h = time_h  # raw string e.g. "0.000000"
            errors[-1]._type_number = type_number
            errors[-1]._raw_unit = raw_unit  # raw string e.g. "Not applicable or not available"
            errors[-1]._raw_type = raw_type  # raw string e.g. "Not applicable or not available"
            errors[-1]._reported_info = reported_info  # e.g. "unit number 8"

        return errors

    def _parse_lst_file(self, lst_content: str) -> List[TRNSYSError]:
        """Parse TRNSYS lst file for errors."""
        errors = []

        if not lst_content:
            return errors

        # Pattern for *** errors
        star_error_pattern = re.compile(r'\*\*\*\s*(.+?)(?=\n|\Z)', re.IGNORECASE)
        for match in star_error_pattern.finditer(lst_content):
            msg = match.group(1).strip()
            if msg and ("error" in msg.lower() or "warning" in msg.lower()):
                severity = "error" if "error" in msg.lower() else "warning"
                errors.append(TRNSYSError(
                    error_type="lst_error",
                    message=msg[:500],
                    severity=severity,
                    source="lst",
                ))

        # Pattern for line number errors
        line_error_pattern = re.compile(r'line\s*(\d+)[:\s]*(.+?)(?=\n|\Z)', re.IGNORECASE)
        for match in line_error_pattern.finditer(lst_content):
            line_num = int(match.group(1))
            msg = match.group(2).strip()

            if "error" in msg.lower() or "invalid" in msg.lower():
                errors.append(TRNSYSError(
                    error_type="syntax_error",
                    message=msg[:500],
                    severity="error",
                    line_number=line_num,
                    source="lst",
                ))

        return errors

    def _analyze_errors_with_llm(
        self,
        errors: List,
        dck_content: str,
        log_content: str,
    ) -> ErrorAnalysis:
        """
        Analyze errors using LLM.

        This method uses LLM for error analysis, NOT hardcoded rules.

        Args:
            errors: List of errors (FormatError or TRNSYSError).
            dck_content: DCK file content.
            log_content: Log file content.

        Returns:
            ErrorAnalysis with root cause and suggestions.
        """
        if not self.llm:
            return ErrorAnalysis(
                root_cause="LLM not available for analysis",
                responsible_agents=["validator"],
                fix_suggestions=["Check DCK format manually"],
                priority="high",
            )

        # Build error description
        error_desc = "\n".join([
            f"- {e}" for e in errors[:15]  # Limit to 15 errors
        ])

        # Build prompt
        system_prompt = """You are a TRNSYS DCK file error analysis expert. Analyze the given error information and determine the root cause and repair suggestions.

[Available Agents]
- planner: system planning
- component_selector: component selection
- parameter_configurator: parameter configuration
- connection_generator: connection generation
- equation_generator: equation generation
- layout_designer: layout design
- validator: validation and assembly

[Output Format]
Return the analysis result in JSON format:
```json
{
    "root_cause": "Root cause of the error",
    "responsible_agents": ["List of agents responsible for the repair"],
    "fix_suggestions": ["List of specific repair suggestions"],
    "priority": "high/medium/low",
    "affected_units": ["List of affected UNIT numbers"]
}
```"""


        user_prompt = f"""Please analyze the following TRNSYS validation errors:

[Error Information]
{error_desc}

[DCK File Excerpt]
{dck_content[:3000]}

[Log Content Excerpt]
{log_content[:1500] if log_content else "No log content available"}

Please analyze the root cause of the errors, determine which agents should be responsible for the repair, and provide specific repair suggestions."""

        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]

            response = self.llm.invoke(messages)
            result_text = response.content.strip()

            # Parse JSON response
            json_match = re.search(r'```json\s*(.*?)\s*```', result_text, re.DOTALL)
            if json_match:
                analysis_dict = json.loads(json_match.group(1))
            else:
                # Try to parse as plain JSON
                analysis_dict = json.loads(result_text)

            return ErrorAnalysis(
                root_cause=analysis_dict.get("root_cause", "Unknown"),
                responsible_agents=analysis_dict.get("responsible_agents", ["validator"]),
                fix_suggestions=analysis_dict.get("fix_suggestions", []),
                priority=analysis_dict.get("priority", "high"),
                affected_units=analysis_dict.get("affected_units", []),
            )

        except Exception as e:
            logger.warning(f"Error analysis parsing failed: {e}")
            return ErrorAnalysis(
                root_cause=f"Analysis failed: {e}",
                responsible_agents=["validator"],
                fix_suggestions=["Review DCK format and TRNSYS log"],
                priority="high",
            )

    def generate_fix_suggestions(
        self,
        validation_result: DCKValidationResult,
        dck_content: str,
    ) -> List[str]:
        """
        Generate fix suggestions based on validation result.

        Args:
            validation_result: The validation result.
            dck_content: DCK file content.

        Returns:
            List of fix suggestions.
        """
        suggestions = []

        # Add suggestions from error analysis
        if validation_result.error_analysis:
            suggestions.extend(validation_result.error_analysis.fix_suggestions)

        # Add suggestions from format errors
        for error in validation_result.format_errors:
            if error.suggestion:
                suggestions.append(error.suggestion)

        # Add generic suggestions based on error types
        error_types = set()
        for error in validation_result.format_errors:
            error_types.add(error.error_type.value if hasattr(error.error_type, 'value') else str(error.error_type))
        for error in validation_result.trnsys_errors:
            error_types.add(error.error_type)

        if "parameters_mismatch" in error_types:
            suggestions.append("Check parameter counts against TMF specifications")

        if "inputs_mismatch" in error_types:
            suggestions.append("Check input counts against TMF specifications")

        if "connection_reference" in error_types:
            suggestions.append("Ensure all referenced UNITs are defined before use")

        if "version_missing" in error_types:
            suggestions.append("Add 'VERSION 18' at the beginning of the file")

        if "end_missing" in error_types:
            suggestions.append("Add 'END' marker at the end of the file")

        # Remove duplicates while preserving order
        seen = set()
        unique_suggestions = []
        for s in suggestions:
            if s not in seen:
                seen.add(s)
                unique_suggestions.append(s)

        return unique_suggestions


# Global validator instance
_validator: Optional[DCKValidator] = None


def get_validator() -> DCKValidator:
    """Get the global validator instance."""
    global _validator
    if _validator is None:
        _validator = DCKValidator()
    return _validator


def reset_validator():
    """Reset the global validator (for testing)."""
    global _validator
    _validator = None


def validate_dck(
    dck_content: str,
    task_id: str = "",
    analyze_errors: bool = True,
) -> DCKValidationResult:
    """
    Convenience function to validate DCK content.

    Args:
        dck_content: The DCK file content.
        task_id: Task identifier.
        analyze_errors: Use LLM to analyze errors.

    Returns:
        DCKValidationResult.
    """
    validator = get_validator()
    return validator.validate(dck_content, task_id, analyze_errors=analyze_errors)
