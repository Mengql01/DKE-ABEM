"""
DCK Snippet Tools for TRNSYS DCK Generator.

Provides tools for searching and comparing DCK snippets from training data.
Implements the "copy when possible" principle.

Key Features:
1. search_dck_snippet: Search for complete DCK snippets of a specific component type
2. get_component_special_fields: Analyze special fields statistics for a component type
"""

from __future__ import annotations

import json
import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.tools import tool
from typing_extensions import Annotated

logger = logging.getLogger(__name__)


def _get_original_data_root_candidates() -> List[Path]:
    """Return possible original TRNSYS data roots across local and Docker layouts."""
    candidates: List[Path] = []

    try:
        from ...core.config import get_settings

        candidates.append(Path(get_settings().original_data_root))
    except Exception as e:
        logger.debug(f"Could not resolve original data root from settings: {e}")

    module_path = Path(__file__).resolve()
    candidates.extend([
        # Current project layout: trnsys_dck_generator/original_trnsys_data
        module_path.parents[3] / "original_trnsys_data",
        # Legacy workspace layout: <workspace>/original_trnsys_data
        module_path.parents[4] / "original_trnsys_data",
    ])

    unique_candidates: List[Path] = []
    seen = set()
    for candidate in candidates:
        candidate = candidate.expanduser()
        if candidate in seen:
            continue
        seen.add(candidate)
        unique_candidates.append(candidate)

    return unique_candidates


def _get_training_json_path(original_data_root: Path) -> Path:
    """Return the parsed training DCK JSON path for an original data root."""
    return original_data_root / "parsed_206_dck_files" / "train_set_dck_parsed.json"


def _extract_global_type_usage(training_data: Dict[str, Any]) -> Dict[str, Any]:
    """Safely extract non-empty global type usage statistics from parsed DCK data."""
    if not isinstance(training_data, dict):
        return {}

    statistics = training_data.get("statistics")
    if not isinstance(statistics, dict):
        return {}

    global_type_usage = statistics.get("global_type_usage")
    if not isinstance(global_type_usage, dict):
        return {}

    return global_type_usage


@lru_cache(maxsize=None)
def _training_json_has_usage_stats(original_data_root: Path) -> bool:
    """Return True when a root has parsed training data with usage statistics."""
    json_path = _get_training_json_path(original_data_root)
    if not json_path.exists():
        return False

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            training_data = json.load(f)
    except Exception as e:
        logger.debug(f"Could not inspect training JSON at {json_path}: {e}")
        return False

    return bool(_extract_global_type_usage(training_data))


def _resolve_original_data_root() -> Path:
    """Resolve the original TRNSYS data root across local and Docker layouts."""
    candidates = _get_original_data_root_candidates()

    # Prefer roots that have parsed training data, because usage ranking depends on it.
    for candidate in candidates:
        if _training_json_has_usage_stats(candidate):
            return candidate

    # Fall back to roots with raw DCK files for snippet-based tools.
    for candidate in candidates:
        if (candidate / "206_dck_files" / "train_set").exists():
            return candidate

    module_path = Path(__file__).resolve()
    return candidates[0] if candidates else module_path.parents[3] / "original_trnsys_data"


ORIGINAL_DATA_ROOT = _resolve_original_data_root()

# Path to training DCK files
TRAINING_DCK_DIR = ORIGINAL_DATA_ROOT / "206_dck_files" / "train_set"
TRAINING_JSON_PATH = _get_training_json_path(ORIGINAL_DATA_ROOT)


@lru_cache(maxsize=1)
def _load_training_json() -> Dict[str, Any]:
    """Load and cache the training data JSON."""
    first_loaded_data: Optional[Dict[str, Any]] = None
    first_loaded_path: Optional[Path] = None
    attempted_paths: List[Path] = []

    for root in _get_original_data_root_candidates():
        json_path = _get_training_json_path(root)
        attempted_paths.append(json_path)

        if not json_path.exists():
            continue

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                training_data = json.load(f)
        except Exception as e:
            logger.warning(f"Could not load training JSON from {json_path}: {e}")
            continue

        if not isinstance(training_data, dict):
            logger.warning(f"Training JSON has invalid top-level type at {json_path}: {type(training_data).__name__}")
            continue

        if first_loaded_data is None:
            first_loaded_data = training_data
            first_loaded_path = json_path

        if _extract_global_type_usage(training_data):
            if json_path != TRAINING_JSON_PATH:
                logger.info(f"Using training JSON with usage statistics from fallback path: {json_path}")
            return training_data

        logger.warning(f"Training JSON has no global_type_usage statistics at {json_path}")

    if first_loaded_data is not None:
        logger.warning(f"Using training JSON from {first_loaded_path}, but global_type_usage statistics are missing.")
        return first_loaded_data

    logger.warning(f"Could not load training JSON from candidate paths: {attempted_paths}")
    return {"results": []}


def _extract_unit_snippets_from_dck(dck_content: str, type_id: str) -> List[str]:
    """
    Extract complete unit snippets for a specific type from DCK content.

    A snippet is the content between two *-------------- separator lines,
    including all special fields like LABELS, External files, etc.

    Args:
        dck_content: Full DCK file content.
        type_id: Component ID from TMF filename (e.g., "Type15-6", "Type65d", "Type56").
            The base type number is extracted for matching against DCK UNIT declarations.

    Returns:
        List of complete snippet strings.
    """
    snippets = []

    # Extract base type number for matching DCK UNIT declarations
    # (e.g., "Type65d" -> "65", "Type15-6" -> "15").
    base_type_num = _get_base_type_number(type_id)
    if not base_type_num:
        return snippets

    # Split by separator lines
    separator_pattern = r'\*-{10,}'
    parts = re.split(separator_pattern, dck_content)

    for i, part in enumerate(parts):
        # Check if this part contains a UNIT declaration for the target type
        unit_pattern = rf'UNIT\s+\d+\s+TYPE\s+{base_type_num}\b'
        if re.search(unit_pattern, part, re.IGNORECASE):
            if _has_variant_qualifier(type_id) and not _snippet_matches_exact_type_id(part, type_id):
                continue

            # Clean up the snippet
            snippet = part.strip()
            if snippet:
                # Add back the separator at the end
                snippet = snippet + "\n*------------------------------------------------------------------------------"
                snippets.append(snippet)

    return snippets


def _normalize_type_id(type_id: str) -> str:
    """Normalize component IDs to TMF filename form without the .tmf suffix."""
    normalized = str(type_id).strip()
    if normalized.lower().endswith(".tmf"):
        normalized = normalized[:-4]
    if not normalized.lower().startswith("type"):
        normalized = f"Type{normalized}"

    match = re.match(r"^type(.+)$", normalized, re.IGNORECASE)
    return f"Type{match.group(1)}" if match else normalized


def _get_base_type_number(type_id: str) -> Optional[str]:
    """Extract the numeric UNIT TYPE value from a component ID."""
    base_match = re.match(r"^Type(\d+)", _normalize_type_id(type_id), re.IGNORECASE)
    return base_match.group(1) if base_match else None


def _has_variant_qualifier(type_id: str) -> bool:
    """Return True when a component ID is more specific than Type<number>."""
    return not re.fullmatch(r"Type\d+", _normalize_type_id(type_id), re.IGNORECASE)


def _extract_tmf_basename(model_path: str) -> str:
    """Extract a TMF filename from a DCK model path line."""
    cleaned = model_path.strip().strip('"')
    basename_match = re.search(r'([^\\/"]+\.tmf)\b', cleaned, re.IGNORECASE)
    if basename_match:
        return basename_match.group(1)
    return re.split(r"[\\/]", cleaned)[-1].strip().strip('"')


def _snippet_matches_exact_type_id(snippet: str, type_id: str) -> bool:
    """
    Check whether a coarse UNIT TYPE snippet belongs to the requested TMF variant.

    TRNSYS DCK UNIT declarations use the base numeric type (for example TYPE 15),
    while TMF variants are stored in metadata such as Type15-6.tmf.
    """
    normalized_type_id = _normalize_type_id(type_id)
    expected_tmf = f"{normalized_type_id}.tmf".lower()

    for model_path in re.findall(r"\*\$MODEL\s+([^\r\n]+)", snippet, re.IGNORECASE):
        if _extract_tmf_basename(model_path).lower() == expected_tmf:
            return True

    unit_name_match = re.search(r"\*\$UNIT_NAME\s+([^\r\n]+)", snippet, re.IGNORECASE)
    if unit_name_match:
        return unit_name_match.group(1).strip().lower() == normalized_type_id.lower()

    return False


def _get_all_training_snippets(type_id: str) -> List[Dict[str, Any]]:
    """
    Get all training snippets for a specific type from all training DCK files.

    Args:
        type_id: Type ID to search for.

    Returns:
        List of dicts with 'filename', 'snippet', and 'unit_name'.
    """
    results = []

    if not TRAINING_DCK_DIR.exists():
        logger.warning(f"Training DCK directory not found: {TRAINING_DCK_DIR}")
        return results

    for dck_file in TRAINING_DCK_DIR.glob("*.dck"):
        try:
            content = dck_file.read_text(encoding='utf-8', errors='ignore')
            snippets = _extract_unit_snippets_from_dck(content, type_id)

            for snippet in snippets:
                # Extract unit name from snippet
                name_match = re.search(r'\*\$UNIT_NAME\s+(\S+)', snippet)
                unit_name = name_match.group(1) if name_match else "Unknown"

                results.append({
                    "filename": dck_file.name,
                    "snippet": snippet,
                    "unit_name": unit_name,
                })
        except Exception as e:
            logger.debug(f"Error reading {dck_file}: {e}")

    return results


def _analyze_special_fields(snippets: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze special fields in snippets.

    Args:
        snippets: List of snippet dicts.

    Returns:
        Statistics about special fields.
    """
    total = len(snippets)
    if total == 0:
        return {"total": 0}

    has_external_files = 0
    has_labels = 0
    has_parameters = 0
    has_inputs = 0

    labels_examples = []
    external_files_examples = []

    for item in snippets:
        snippet = item["snippet"]

        # Check for External files
        if "*** External files" in snippet or re.search(r'ASSIGN\s+"[^"]+"\s+\d+', snippet):
            has_external_files += 1
            # Extract ASSIGN line
            assign_match = re.search(r'ASSIGN\s+"([^"]+)"\s+(\d+)', snippet)
            if assign_match:
                external_files_examples.append({
                    "path": assign_match.group(1),
                    "logical_unit": int(assign_match.group(2)),
                })

        # Check for LABELS
        labels_match = re.search(r'LABELS\s+(\d+)\s*\n((?:"[^"]*"\s*\n?)+)', snippet)
        if labels_match:
            has_labels += 1
            count = int(labels_match.group(1))
            labels_text = labels_match.group(2)
            label_values = re.findall(r'"([^"]*)"', labels_text)
            labels_examples.append({
                "count": count,
                "values": label_values,
            })

        # Check for PARAMETERS
        if re.search(r'PARAMETERS\s+\d+', snippet):
            has_parameters += 1

        # Check for INPUTS
        if re.search(r'INPUTS\s+\d+', snippet):
            has_inputs += 1

    return {
        "total": total,
        "has_external_files": has_external_files,
        "external_files_pct": round(has_external_files / total * 100, 1),
        "has_labels": has_labels,
        "labels_pct": round(has_labels / total * 100, 1),
        "has_parameters": has_parameters,
        "parameters_pct": round(has_parameters / total * 100, 1),
        "has_inputs": has_inputs,
        "inputs_pct": round(has_inputs / total * 100, 1),
        "labels_examples": labels_examples[:3],
        "external_files_examples": external_files_examples[:3],
    }


@tool
def search_dck_snippet(
    type_id: Annotated[str, "Component Type ID, e.g., 'Type15-6', 'Type65d', 'Type56'"],
    limit: Annotated[int, "Maximum number of snippets to return"] = 3,
) -> str:
    """
    Search for complete DCK snippets of a specific component type from training data.

    Returns the raw DCK text between two *------------------------------------------------------------------------------
    separator lines, including all special fields like LABELS, External files, etc.

    Use this tool to:
    1. See the correct format for a component type
    2. Check what special fields (LABELS, External files) are typically used
    3. Compare with generated DCK to find missing fields

    Args:
        type_id: Component Type ID (e.g., "Type15-6", "Type65d", "Type56")
        limit: Maximum number of snippets to return (default 3)

    Returns:
        Formatted string with DCK snippets from training data.
    """
    snippets = _get_all_training_snippets(type_id)

    if not snippets:
        return f"No training snippets found for {type_id}. Check if the type ID is correct."

    # Limit results
    snippets = snippets[:limit]

    # Format output
    output_lines = [f"=== Training DCK Snippets for {type_id} ==="]
    output_lines.append(f"Found {len(snippets)} snippet(s) (showing up to {limit}):\n")

    for i, item in enumerate(snippets, 1):
        output_lines.append(f"--- Snippet {i} (from {item['filename']}, unit: {item['unit_name']}) ---")
        output_lines.append(item["snippet"])
        output_lines.append("")

    return "\n".join(output_lines)


@tool
def get_component_special_fields(
    type_id: Annotated[str, "Component ID from TMF filename, e.g., 'Type15-6', 'Type65d', 'Type56'"],
) -> str:
    """
    Analyze special fields statistics for a component type from training data.

    CRITICAL: type_id MUST be the exact TMF filename without .tmf extension.
    Use "Type15-6" not "Type15", "Type65d" not "Type65".

    Returns statistics about what special fields (LABELS, External files, PARAMETERS, INPUTS)
    are typically present for this component type in the training DCK files.

    Use this tool to:
    1. Determine if a component type typically has LABELS
    2. Determine if a component type typically has External files
    3. Get example values for these special fields

    Args:
        type_id: Component ID (TMF filename without .tmf, e.g., "Type15-6", "Type65d", "Type56")

    Returns:
        Formatted string with special fields statistics.
    """
    snippets = _get_all_training_snippets(type_id)
    stats = _analyze_special_fields(snippets)

    if stats["total"] == 0:
        return f"No training data found for {type_id}."

    output_lines = [f"=== Special Fields Analysis for {type_id} ==="]
    output_lines.append(f"\nTotal instances in training data: {stats['total']}")

    # External files
    output_lines.append(f"\nWith External files: {stats['has_external_files']} ({stats['external_files_pct']}%)")
    if stats['external_files_examples']:
        output_lines.append("  Examples:")
        for ex in stats['external_files_examples']:
            output_lines.append(f"    - Path: {ex['path']}, Logical unit: {ex['logical_unit']}")

    # LABELS
    output_lines.append(f"\nWith LABELS: {stats['has_labels']} ({stats['labels_pct']}%)")
    if stats['labels_examples']:
        output_lines.append("  Examples:")
        for ex in stats['labels_examples']:
            output_lines.append(f"    - Count: {ex['count']}, Values: {ex['values']}")

    # Summary
    output_lines.append("\n=== Summary ===")
    if stats['external_files_pct'] > 50:
        output_lines.append(f"This component type USUALLY has External files section")
    elif stats['external_files_pct'] > 0:
        output_lines.append(f"This component type SOMETIMES has External files section")
    else:
        output_lines.append(f"This component type does NOT use External files section")

    if stats['labels_pct'] > 50:
        output_lines.append(f"This component type USUALLY has LABELS block")
    elif stats['labels_pct'] > 0:
        output_lines.append(f"This component type SOMETIMES has LABELS block")
    else:
        output_lines.append(f"This component type does NOT use LABELS block")

    return "\n".join(output_lines)


# DEPRECATED: compare_dck_snippet tool is no longer used in testing
# This tool was never invoked during actual testing, while other DCK snippet tools
# (search_dck_snippet, get_component_special_fields, rank_components_by_usage) are used multiple times.
# Kept as commented code for reference, but removed from active toolset.
#
# @tool
# def compare_dck_snippet(
#     generated_snippet: Annotated[str, "The generated DCK snippet to compare"],
#     type_id: Annotated[str, "Component ID from TMF filename for reference lookup, e.g., 'Type15-6', 'Type65d', 'Type56'"],
# ) -> str:
#     """
#     Compare a generated DCK snippet with training data snippets.
#
#     CRITICAL: type_id MUST be the exact TMF filename without .tmf extension.
#     Use "Type15-6" not "Type15", "Type65d" not "Type65".
#
#     Identifies missing or extra fields by comparing the generated snippet
#     with reference snippets from training data.
#
#     Args:
#         generated_snippet: The DCK snippet generated by the system
#         type_id: Component ID (TMF filename without .tmf, e.g., "Type15-6", "Type65d", "Type56")
#
#     Returns:
#         Analysis of differences and suggestions for fixes.
#     """
#     # Get reference snippets
#     ref_snippets = _get_all_training_snippets(type_id)
#     if not ref_snippets:
#         return f"No reference snippets found for {type_id}."
#
#     # Analyze the generated snippet
#     gen_has_external = "*** External files" in generated_snippet or re.search(r'ASSIGN\s+"[^"]+"\s+\d+', generated_snippet)
#     gen_has_labels = re.search(r'LABELS\s+\d+', generated_snippet)
#     gen_has_params = re.search(r'PARAMETERS\s+\d+', generated_snippet)
#     gen_has_inputs = re.search(r'INPUTS\s+\d+', generated_snippet)
#
#     # Analyze reference snippets
#     stats = _analyze_special_fields(ref_snippets)
#
#     # Build comparison report
#     output_lines = [f"=== DCK Snippet Comparison for {type_id} ==="]
#     output_lines.append(f"\nReference data: {stats['total']} training snippets")
#
#     issues = []
#     suggestions = []
#
#     # Check External files
#     if stats['external_files_pct'] > 50 and not gen_has_external:
#         issues.append("MISSING: External files section (present in {:.0f}% of training data)".format(stats['external_files_pct']))
#         if stats['external_files_examples']:
#             ex = stats['external_files_examples'][0]
#             suggestions.append(f"Add External files section with ASSIGN statement (example logical unit: {ex['logical_unit']})")
#     elif stats['external_files_pct'] == 0 and gen_has_external:
#         issues.append("EXTRA: External files section (not present in training data)")
#
#     # Check LABELS
#     if stats['labels_pct'] > 50 and not gen_has_labels:
#         issues.append("MISSING: LABELS block (present in {:.0f}% of training data)".format(stats['labels_pct']))
#         if stats['labels_examples']:
#             ex = stats['labels_examples'][0]
#             suggestions.append(f"Add LABELS block with {ex['count']} labels: {ex['values']}")
#     elif stats['labels_pct'] == 0 and gen_has_labels:
#         issues.append("EXTRA: LABELS block (not present in training data)")
#
#     # Report
#     if issues:
#         output_lines.append("\n=== Issues Found ===")
#         for issue in issues:
#             output_lines.append(f"  - {issue}")
#
#         output_lines.append("\n=== Suggestions ===")
#         for sug in suggestions:
#             output_lines.append(f"  - {sug}")
#     else:
#         output_lines.append("\nNo major structural differences found.")
#
#     # Show generated vs reference
#     output_lines.append("\n=== Generated Snippet Structure ===")
#     output_lines.append(f"  Has PARAMETERS: {'is' if gen_has_params else 'No'}")
#     output_lines.append(f"  Has INPUTS: {'is' if gen_has_inputs else 'No'}")
#     output_lines.append(f"  Has External files: {'is' if gen_has_external else 'No'}")
#     output_lines.append(f"  Has LABELS: {'is' if gen_has_labels else 'No'}")
#
#     output_lines.append("\n=== Training Data Statistics ===")
#     output_lines.append(f"  With PARAMETERS: {stats['parameters_pct']}%")
#     output_lines.append(f"  With INPUTS: {stats['inputs_pct']}%")
#     output_lines.append(f"  With External files: {stats['external_files_pct']}%")
#     output_lines.append(f"  With LABELS: {stats['labels_pct']}%")
#
#     return "\n".join(output_lines)


# === Component Usage Frequency Tool ===

@tool
def rank_components_by_usage(
    candidate_groups_json: Annotated[str, "JSON string of candidate groups. Format: {\"category\": [\"TypeA\", \"TypeB\"], ...}. Example: '{\"weather\": [\"Type15-6\", \"Type15-2\"], \"pump\": [\"Type114\", \"Type110\"]}'"],
) -> str:
    """
    Rank candidate components by their usage frequency in training DCK files.

    For each category, looks up every candidate's usage count from the
    global_type_usage statistics in the training dataset (206 DCK files),
    and returns the best component (highest usage count) for each category.

    Components not found in global_type_usage have a usage count of 0.

    Args:
        candidate_groups_json: JSON string mapping category names to lists of
            candidate component IDs (TMF filename without .tmf).
            Example: '{"weather": ["Type15-6", "Type15-2"], "pump": ["Type114", "Type110"]}'

    Returns:
        Formatted string showing usage counts for all candidates and the
        recommended best component for each category.
    """
    try:
        candidate_groups = json.loads(candidate_groups_json)
    except json.JSONDecodeError as e:
        return f"ERROR: Invalid JSON input: {e}. Expected format: {{\"category\": [\"TypeA\", \"TypeB\"]}}"

    if not isinstance(candidate_groups, dict):
        return "ERROR: Input must be a JSON object mapping category names to lists of component IDs."

    # Load usage data from training JSON
    training_data = _load_training_json()
    global_type_usage = training_data.get("statistics", {}).get("global_type_usage", {})

    if not global_type_usage:
        return "WARNING: Could not load component usage statistics from training data. All usage counts default to 0."

    output_lines = ["=== Component Usage Frequency Ranking ===\n"]

    for category, candidates in candidate_groups.items():
        if not isinstance(candidates, list) or not candidates:
            output_lines.append(f"Category '{category}': No valid candidates provided.\n")
            continue

        output_lines.append(f"Category: {category}")

        # Look up usage count for each candidate
        usage_counts = []
        for comp_id in candidates:
            # Extract base type number from TMF filename for lookup
            # e.g., "Type15-6" -> "Type15", "Type65d" -> "Type65d", "Type56" -> "Type56"
            # This handles the mismatch between precise TMF filenames (Type15-6)
            # and global_type_usage keys (Type15) in training data statistics
            base_match = re.match(r'(Type\d+)', comp_id, re.IGNORECASE)
            lookup_key = base_match.group(1) if base_match else comp_id

            # Try exact match first, then fallback to base type
            count = global_type_usage.get(comp_id, 0)
            if count == 0 and lookup_key != comp_id:
                count = global_type_usage.get(lookup_key, 0)

            usage_counts.append((comp_id, count))

        # Sort by usage count descending
        usage_counts.sort(key=lambda x: x[1], reverse=True)

        for comp_id, count in usage_counts:
            output_lines.append(f"  - {comp_id}: {count} occurrences in training DCKs")

        # Recommend the best
        best_id, best_count = usage_counts[0]
        output_lines.append(f"  >> RECOMMENDED: {best_id} (usage count: {best_count})")
        output_lines.append("")

    return "\n".join(output_lines)


# Convenience functions for programmatic use (not as tools)

def get_dck_snippet_tools() -> List:
    """Get all DCK snippet tools."""
    # DEPRECATED: compare_dck_snippet removed from active toolset (never used in testing)
    return [search_dck_snippet, get_component_special_fields, rank_components_by_usage]


def get_dck_snippet_tools_for_agent(agent_name: str) -> List:
    """Get DCK snippet tools appropriate for a specific agent."""
    # These tools are useful for validator and parameter_configurator
    if agent_name in ["validator", "parameter_configurator", "component_selector"]:
        return get_dck_snippet_tools()
    return []


def get_special_fields_for_type(type_id: str) -> Dict[str, Any]:
    """
    Get special fields information for a type (programmatic use).

    Returns dict with:
    - has_external_files: bool
    - has_labels: bool
    - default_labels: List[str] (if applicable)
    - default_logical_unit: int (if applicable)
    """
    snippets = _get_all_training_snippets(type_id)
    stats = _analyze_special_fields(snippets)

    result = {
        "has_external_files": stats.get("external_files_pct", 0) > 50,
        "has_labels": stats.get("labels_pct", 0) > 50,
        "external_files_pct": stats.get("external_files_pct", 0),
        "labels_pct": stats.get("labels_pct", 0),
        "default_labels": [],
        "default_logical_unit": 30,
    }

    # Get default labels from examples
    if stats.get("labels_examples"):
        result["default_labels"] = stats["labels_examples"][0].get("values", [])

    # Get default logical unit from examples
    if stats.get("external_files_examples"):
        result["default_logical_unit"] = stats["external_files_examples"][0].get("logical_unit", 30)

    return result


def clear_dck_snippet_cache():
    """Clear the training data cache."""
    _training_json_has_usage_stats.cache_clear()
    _load_training_json.cache_clear()
