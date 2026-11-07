"""
LangSmith Trace Filter for TRNSYS DCK Generator.

Uses the official LangSmith Client(hide_inputs=fn, hide_outputs=fn) mechanism
to filter large state fields from traces. This approach:
- Returns a NEW filtered dict (never mutates the original State)
- Filtering happens inside the LangSmith Client at send-time
- Completely safe for LangGraph State integrity

Official documentation:
    https://docs.langchain.com/langsmith/mask-inputs-outputs

Usage:
    from src.core.trace_filter import get_trace_filter_callbacks
    config = {"callbacks": get_trace_filter_callbacks(), ...}
    graph.ainvoke(state, config=config)
"""

from __future__ import annotations

import logging
from typing import Any, List

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration: fields to hide from LangSmith traces
# ---------------------------------------------------------------------------

HIDDEN_STATE_FIELDS: List[str] = [
    "agent_last_outputs",
    "first_round_agent_raw_outputs",
    # "validation_file_versions",  # Moved to fine-grained filtering
    "dck_content",
    "retrieved_documents",
    "similar_dcks",
]

# Fields within validation_file_versions items to filter
# (preserves path fields: dck_path, log_path, lst_path, version_tag)
VALIDATION_FILE_CONTENT_FIELDS: List[str] = [
    "dck_content",
    "log_content",
    "lst_content",
]


# ---------------------------------------------------------------------------
# Filtering logic (pure function, never mutates input)
# ---------------------------------------------------------------------------

def _summarize_value(key: str, value: Any) -> str:
    """Create a short summary string for a hidden field value."""
    if isinstance(value, str):
        return f"<filtered: str, {len(value)} chars>"
    if isinstance(value, dict):
        return f"<filtered: dict, {len(value)} keys>"
    if isinstance(value, list):
        return f"<filtered: list, {len(value)} items>"
    return f"<filtered: {type(value).__name__}>"


def _filter_validation_file_versions(versions: List[dict], reference_content: str = None) -> List[dict]:
    """
    Filter large content fields from validation_file_versions items.

    This function processes each item in the validation_file_versions list
    and replaces large content fields (dck_content, log_content, lst_content)
    with summary strings, while preserving path fields (dck_path, log_path,
    lst_path, version_tag).

    For dck_content, if it's an empty string and reference_content is provided,
    the summary will reflect the actual content size from reference_content.

    Args:
        versions: List of file version dictionaries.
        reference_content: Optional reference dck_content from top-level state
                          to calculate actual size when the field is empty.

    Returns:
        Filtered list with content fields replaced by summaries.
    """
    if not isinstance(versions, list):
        return versions

    filtered_versions = []
    for item in versions:
        if not isinstance(item, dict):
            filtered_versions.append(item)
            continue

        filtered_item = dict(item)
        for field in VALIDATION_FILE_CONTENT_FIELDS:
            if field in filtered_item:
                val = filtered_item[field]
                if val is not None and val != "" and val != [] and val != {}:
                    filtered_item[field] = _summarize_value(field, val)
                # For empty dck_content, use reference content size if available
                elif val == "" and field == "dck_content" and reference_content:
                    filtered_item[field] = f"<filtered: str, {len(reference_content)} chars>"
                # Other empty fields
                elif val == "":
                    filtered_item[field] = "<filtered: empty>"

        filtered_versions.append(filtered_item)

    return filtered_versions


def filter_state_for_tracing(data: Any) -> Any:
    """
    Filter large fields from a state dictionary for LangSmith tracing.

    This is a PURE FUNCTION: it returns a new dict with large fields replaced
    by summary strings. The original data is NEVER modified.

    Args:
        data: The inputs or outputs dictionary to filter.

    Returns:
        Filtered copy if large fields found, otherwise the original unchanged.
    """
    if not isinstance(data, dict):
        return data

    has_hidden = False
    for field in HIDDEN_STATE_FIELDS:
        val = data.get(field)
        if val is not None and val != "" and val != [] and val != {}:
            has_hidden = True
            break

    # Check if validation_file_versions needs fine-grained filtering
    if "validation_file_versions" in data:
        versions = data.get("validation_file_versions")
        if versions:
            has_hidden = True

    if not has_hidden:
        return data

    filtered = dict(data)

    # Apply general field filtering
    for field in HIDDEN_STATE_FIELDS:
        if field in filtered:
            val = filtered[field]
            if val is not None and val != "" and val != [] and val != {}:
                filtered[field] = _summarize_value(field, val)

    # Apply fine-grained filtering for validation_file_versions
    if "validation_file_versions" in filtered:
        # Pass top-level dck_content as reference for size calculation
        reference_dck_content = data.get("dck_content", None)
        filtered["validation_file_versions"] = _filter_validation_file_versions(
            filtered["validation_file_versions"],
            reference_content=reference_dck_content
        )

    return filtered


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_trace_filter_callbacks() -> List[Any]:
    """
    Create a LangChainTracer with a filtered LangSmith Client.

    Uses the official LangSmith Client(hide_inputs=fn, hide_outputs=fn)
    mechanism. The filter functions receive the original data and return
    a filtered COPY — the original State is never mutated.

    Since the returned list contains a LangChainTracer instance, LangGraph's
    auto-tracing will NOT inject a second default tracer (it checks
    `isinstance(handler, LangChainTracer)` before adding one).

    Returns:
        List containing a single LangChainTracer with filtering enabled.

    Usage:
        config = {"callbacks": get_trace_filter_callbacks(), ...}
        graph.ainvoke(state, config=config)
    """
    try:
        from langsmith import Client
        from langchain_core.tracers import LangChainTracer

        client = Client(
            hide_inputs=filter_state_for_tracing,
            hide_outputs=filter_state_for_tracing,
        )
        tracer = LangChainTracer(client=client)
        return [tracer]

    except Exception as e:
        logger.warning(
            "Failed to create trace filter callbacks: %s. "
            "Tracing will proceed without filtering.",
            e,
        )
        return []
