"""
In-process cancellation registry for LangGraph runs.

When LangGraph Runtime cancels a run (via Studio Cancel button), the runtime
marks the run as "interrupted" in its internal store. Synchronous node code
running in a thread pool cannot receive the asyncio CancelledError directly.

This module provides two detection paths:
1. In-memory registry (mark_run_cancelled / is_run_cancelled) — fast, set by
   any code that catches CancelledError.
2. Local Runtime API poll (check_run_status_via_api) — authoritative, queries
   the runtime's own run record via HTTP GET on localhost:2024.
"""

import logging
import threading
from typing import Optional, Set

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_cancelled_run_ids: Set[str] = set()

RUNTIME_BASE_URL = "http://127.0.0.1:2024"


def mark_run_cancelled(run_id: str) -> None:
    with _lock:
        _cancelled_run_ids.add(run_id)
    logger.info("Cancellation registry: marked run %s as cancelled", run_id)


def is_run_cancelled(run_id: str) -> bool:
    if not run_id:
        return False
    with _lock:
        return run_id in _cancelled_run_ids


def clear_cancelled_run(run_id: str) -> None:
    with _lock:
        _cancelled_run_ids.discard(run_id)


def check_run_status_via_api(thread_id: str, run_id: str) -> bool:
    """Query local LangGraph Runtime API to check if a run has been cancelled.

    Returns True if the run status is 'interrupted', False otherwise.
    On any error (network, parse, etc.) returns False to avoid blocking.
    """
    if not thread_id or not run_id:
        return False
    try:
        import urllib.request
        import json

        url = f"{RUNTIME_BASE_URL}/threads/{thread_id}/runs/{run_id}"
        req = urllib.request.Request(url, method="GET")
        req.add_header("Accept", "application/json")
        with urllib.request.urlopen(req, timeout=2) as resp:
            data = json.loads(resp.read().decode())
            status = data.get("status", "")
            if status == "interrupted":
                mark_run_cancelled(run_id)
                logger.info("API check: run %s is interrupted", run_id)
                return True
    except Exception:
        pass
    return False
