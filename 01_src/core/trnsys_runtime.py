"""
TRNSYS Runtime Compatibility Layer.

Provides cross-platform support for running TrnEXE64.exe:
- Windows: Direct execution
- Linux (Docker): Wine-based execution with path translation
- Host mode: Execute on Windows host via HTTP API (Docker container only)

This module abstracts the differences between Windows native execution,
Wine-based execution in Linux containers, and Windows host execution,
allowing the codebase to work seamlessly in all environments with minimal changes.
"""

import os
import sys
from pathlib import Path
from typing import List, Union, Dict, Any, Optional

# Import requests for host mode HTTP communication
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


def is_wine_mode() -> bool:
    """
    Check if running in Wine mode (Linux container with TRNSYS_USE_WINE=1).

    Returns:
        True if running in Wine mode, False otherwise

    Examples:
        >>> os.environ['TRNSYS_USE_WINE'] = '1'
        >>> is_wine_mode()  # On Linux
        True
        >>> is_wine_mode()  # On Windows
        False
    """
    return os.name != "nt" and os.getenv("TRNSYS_USE_WINE", "0") == "1"


def is_host_mode() -> bool:
    """
    Check if running in Windows host execution mode.

    In host mode, TRNSYS execution is delegated to a Windows host service
    via HTTP API, avoiding Wine compatibility issues in Docker containers.

    Returns:
        True if TRNSYS_USE_HOST=1, False otherwise

    Examples:
        >>> os.environ['TRNSYS_USE_HOST'] = '1'
        >>> is_host_mode()
        True
    """
    return os.getenv("TRNSYS_USE_HOST", "0") == "1"


def execute_trnsys_on_host(
    dck_path: Union[str, Path],
    *args: str,
    timeout: Optional[int] = None
) -> Dict[str, Any]:
    """
    Execute TRNSYS on Windows host via HTTP API.

    This function sends an HTTP request to the Windows host service,
    which executes TrnEXE64.exe natively and returns the results.
    The DCK file and output files (log/lst) are shared via Docker bind mount.

    Args:
        dck_path: Path to .dck file (container path format)
        *args: Additional arguments (e.g., "/h" for hidden mode)
        timeout: Request timeout in seconds. If None (default), the value is
            read from performance_config.trnsys_validation_timeout, keeping the
            execution timeout configurable from a single place.

    Returns:
        Dictionary with execution results:
        {
            "success": bool,
            "exit_code": int,
            "log_path": str,
            "lst_path": str,
            "stdout": str,
            "stderr": str
        }

    Raises:
        RuntimeError: If service is unavailable or execution fails
        ImportError: If requests library is not available

    Examples:
        >>> result = execute_trnsys_on_host("/deps/trnsys_dck_generator/validation_runs/test.dck", "/h")
        >>> print(result["success"])
        True
        >>> print(result["log_path"])
        /deps/trnsys_dck_generator/validation_runs/test.log
    """
    if not REQUESTS_AVAILABLE:
        raise ImportError(
            "requests library is required for host mode execution. "
            "Install it with: pip install requests"
        )

    # Resolve the execution timeout from performance_config when not explicitly
    # provided, so that the timeout is configurable from a single source of truth.
    if timeout is None:
        try:
            from .performance_config import get_performance_config
            timeout = get_performance_config().trnsys_validation_timeout
        except Exception:
            # Fallback to a safe default if config cannot be loaded.
            timeout = 6000

    service_url = os.getenv("TRNSYS_HOST_SERVICE_URL", "http://host.docker.internal:5678")

    # Prepare request
    payload = {
        "dck_path": str(dck_path),
        "args": list(args) if args else ["/h"]
    }

    try:
        # Send execution request
        response = requests.post(
            f"{service_url}/execute",
            json=payload,
            timeout=timeout
        )
        response.raise_for_status()

        result = response.json()

        # Note: Do not raise exception here even if success is False
        # The caller (validator.py) needs to read log/lst files to parse TRNSYS errors
        # TRNSYS errors are in the log file, not in stdout/stderr
        # Let the caller decide how to handle the result

        return result

    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            f"Cannot connect to TRNSYS execution service at {service_url}. "
            "Please ensure the service is running on Windows host. "
            "Start it with: start_trnsys_service.bat"
        )
    except requests.exceptions.Timeout:
        raise RuntimeError(
            f"TRNSYS execution timeout ({timeout}s). "
            "The simulation may be too complex or the system is overloaded."
        )
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"HTTP request failed: {str(e)}")


def build_trnsys_cmd(
    exe_path: Union[str, Path],
    dck_path: Union[str, Path],
    *args: str
) -> List[str]:
    """
    Build TRNSYS execution command with Wine/Host support.

    Automatically prepends 'wine' command in Wine mode and translates
    the executable path to Wine-compatible format.

    NOTE: In host mode, this function should not be called.
    Use execute_trnsys_on_host() instead.

    Args:
        exe_path: Path to TrnEXE64.exe (Windows or Linux format)
        dck_path: Path to .dck file
        *args: Additional arguments (e.g., "/h" for hidden mode)

    Returns:
        Command list ready for subprocess.run() or subprocess.Popen()

    Raises:
        RuntimeError: If called in host mode

    Examples:
        Windows mode:
            build_trnsys_cmd("D:\\TRNSYS18_04\\Exe\\TrnEXE64.exe", "test.dck", "/h")
            -> ["D:\\TRNSYS18_04\\Exe\\TrnEXE64.exe", "test.dck", "/h"]

        Wine mode:
            build_trnsys_cmd("D:\\TRNSYS18_04\\Exe\\TrnEXE64.exe", "test.dck", "/h")
            -> ["xvfb-run", "-a", "wine", "/mnt/trnsys/TRNSYS18_04/Exe/TrnEXE64.exe", "Z:\\...", "/h"]
    """
    exe_str = str(exe_path)
    dck_str = str(dck_path)

    if is_host_mode():
        # Host mode: execution happens via HTTP API, not subprocess
        # This function should not be called in host mode
        raise RuntimeError(
            "build_trnsys_cmd() should not be called in host mode. "
            "Use execute_trnsys_on_host() instead."
        )
    elif is_wine_mode():
        # Wine mode: existing logic
        # Translate Windows path to Wine path
        exe_wine = translate_path_to_wine(exe_str)
        # Translate dck_path to Wine Windows format (Z:\...)
        # so that TrnEXE64.exe (a Windows program) can locate the file
        dck_wine = translate_path_to_wine_windows(dck_str)
        # Use xvfb-run to provide a virtual X display for Wine
        return ["xvfb-run", "-a", "wine", exe_wine, dck_wine, *args]
    else:
        # Windows native execution
        return [exe_str, dck_str, *args]


def translate_path_to_wine(windows_path: str) -> str:
    """
    Translate Windows absolute path to Wine-compatible Linux path.

    Converts Windows-style paths (D:\\...) to the mounted Linux path
    (/mnt/trnsys/...) used in the Docker container.

    Args:
        windows_path: Windows absolute path (e.g., D:\\TRNSYS18_04\\Exe\\TrnEXE64.exe)

    Returns:
        Wine-compatible Linux path (e.g., /mnt/trnsys/TRNSYS18_04/Exe/TrnEXE64.exe)

    Examples:
        >>> translate_path_to_wine("D:\\\\TRNSYS18_04\\\\Exe\\\\TrnEXE64.exe")
        '/mnt/trnsys/TRNSYS18_04/Exe/TrnEXE64.exe'

        >>> translate_path_to_wine("D:/TRNSYS18_04/Weather/file.tm2")
        '/mnt/trnsys/TRNSYS18_04/Weather/file.tm2'
    """
    # Normalize path separators to forward slashes
    normalized = windows_path.replace("\\", "/")

    # Idempotency: if already a Linux mount path, return as-is
    if normalized.startswith("/mnt/trnsys/"):
        return normalized

    # Remove drive letter (D:) if present
    if len(normalized) >= 2 and normalized[1] == ":":
        normalized = normalized[2:]  # Remove "D:"

    # Remove leading slash if present
    normalized = normalized.lstrip("/")

    # Prepend mount point
    return f"/mnt/trnsys/{normalized}"


def translate_path_to_wine_windows(linux_path: str) -> str:
    """
    Translate a Linux container path to Wine Windows format (Z:\\...).

    Wine maps the root filesystem / to the Z: drive by default.
    This converts paths like /deps/foo/bar.dck to Z:\\deps\\foo\\bar.dck
    so that Windows programs running under Wine can locate the file.

    Args:
        linux_path: A Linux absolute path inside the container.

    Returns:
        Wine-compatible Windows path (e.g., Z:\\deps\\foo\\bar.dck),
        or the original path if it is already in Windows format.
    """
    normalized = linux_path.replace("\\", "/")

    # Already a Windows path (has drive letter like D: or Z:), return as-is
    if len(normalized) >= 2 and normalized[1] == ":":
        return linux_path

    # Linux absolute path -> Z: drive
    if normalized.startswith("/"):
        return "Z:" + normalized.replace("/", "\\")

    # Relative path, return as-is
    return linux_path


def trnsys_exe_exists(exe_path: Union[str, Path]) -> bool:
    """
    Check if TrnEXE64.exe exists (Wine-aware).

    In Wine mode, translates the Windows path to Wine path before checking.
    In Windows mode, checks the path directly.

    Args:
        exe_path: Path to TrnEXE64.exe (Windows or Linux format)

    Returns:
        True if executable exists, False otherwise

    Examples:
        Windows mode:
            trnsys_exe_exists("D:\\TRNSYS18_04\\Exe\\TrnEXE64.exe")
            -> checks D:\\TRNSYS18_04\\Exe\\TrnEXE64.exe

        Wine mode:
            trnsys_exe_exists("D:\\TRNSYS18_04\\Exe\\TrnEXE64.exe")
            -> checks /mnt/trnsys/TRNSYS18_04/Exe/TrnEXE64.exe
    """
    if is_wine_mode():
        # Translate to Wine path and check
        wine_path = translate_path_to_wine(str(exe_path))
        return os.path.isfile(wine_path)
    else:
        # Windows native check
        return os.path.isfile(str(exe_path))
