"""
Connection Pattern Tools for TRNSYS DCK Generator.

Provides tools for querying connection patterns from training DCK data
with PORT NAMES (not just port numbers) resolved via TMF specifications.

This solves the critical problem where LLM copies port numbers from training
data without verifying that the port names/physical quantities match,
leading to incorrect connections (e.g., temperature connected to humidity).

Key Features:
1. search_connection_patterns: Search connection patterns between two component types
2. get_connection_pattern: Get detailed connection pattern for a specific pair
3. find_dcks_with_components: Find DCK files containing specific component combinations
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


# === Data Loading ===

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
        key = str(candidate).lower()
        if key in seen:
            continue
        seen.add(key)
        unique_candidates.append(candidate)

    return unique_candidates


def _get_training_dck_json_path(original_data_root: Path) -> Path:
    """Return the parsed training DCK JSON path for an original data root."""
    return original_data_root / "parsed_206_dck_files" / "train_set_dck_parsed.json"


def _get_tmf_json_path(original_data_root: Path) -> Path:
    """Return the parsed TMF JSON path for an original data root."""
    return original_data_root / "parsed_765_tmf_files.json"


def _parse_data_version(version: str) -> Tuple[int, int]:
    """Parse a data version string such as 'V3.4' into a sortable tuple."""
    match = re.search(r"(\d+)(?:\.(\d+))?", str(version))
    if not match:
        return (0, 0)
    return (int(match.group(1)), int(match.group(2) or 0))


def _score_training_dck_data(training_data: Dict[str, Any]) -> Tuple[int, int, int, int, int]:
    """Score parsed DCK data for connection-pattern completeness."""
    connection_counts = Counter(
        conn.get("connection_type", "")
        for dck_result in training_data.get("results", [])
        for conn in dck_result.get("connections", [])
    )
    complete_connection_types = sum(
        1
        for conn_type in ("equation_to_unit", "unit_to_equation", "equation_to_equation")
        if connection_counts.get(conn_type, 0) > 0
    )
    equation_input_count = sum(
        len(unit.get("inputs", []))
        for dck_result in training_data.get("results", [])
        for unit in dck_result.get("units", [])
        if unit.get("unit_type") == "EQUATIONS"
    )
    version_major, version_minor = _parse_data_version(
        training_data.get("statistics", {}).get("version", "")
    )
    total_connections = (
        training_data.get("statistics", {})
        .get("totals", {})
        .get("total_connections")
        or sum(connection_counts.values())
    )

    return (
        complete_connection_types,
        1 if equation_input_count > 0 else 0,
        version_major,
        version_minor,
        int(total_connections),
    )


@lru_cache(maxsize=1)
def _select_training_dck_source() -> Tuple[Dict[str, Any], Optional[Path]]:
    """Load the best available parsed training DCK JSON and return its path."""
    best_data: Optional[Dict[str, Any]] = None
    best_path: Optional[Path] = None
    best_score: Optional[Tuple[int, int, int, int, int]] = None

    for original_data_root in _get_original_data_root_candidates():
        path = _get_training_dck_json_path(original_data_root)
        if not path.exists():
            continue

        try:
            with open(path, "r", encoding="utf-8") as f:
                training_data = json.load(f)
        except Exception as e:
            logger.warning(f"Could not load training DCK data from {path}: {e}")
            continue

        if not isinstance(training_data, dict):
            logger.warning(
                f"Training DCK data has invalid top-level type at {path}: "
                f"{type(training_data).__name__}"
            )
            continue

        score = _score_training_dck_data(training_data)
        if best_score is None or score > best_score:
            best_data = training_data
            best_path = path
            best_score = score

    if best_data is not None:
        logger.info(f"Using training DCK data from {best_path} with score {best_score}")
        return best_data, best_path

    logger.warning("Could not load any parsed training DCK data from known locations")
    return {"results": []}, None


@lru_cache(maxsize=1)
def _load_training_dck_data() -> Dict[str, Any]:
    """Load and cache the best available training DCK parsed JSON."""
    training_data, _ = _select_training_dck_source()
    return training_data


@lru_cache(maxsize=1)
def _load_tmf_data() -> Dict[str, Any]:
    """Load and cache the TMF parsed JSON for port name resolution."""
    _, training_path = _select_training_dck_source()

    candidate_roots: List[Path] = []
    if training_path is not None:
        candidate_roots.append(training_path.parents[1])
    candidate_roots.extend(_get_original_data_root_candidates())

    seen = set()
    for original_data_root in candidate_roots:
        key = str(original_data_root).lower()
        if key in seen:
            continue
        seen.add(key)

        path = _get_tmf_json_path(original_data_root)
        if not path.exists():
            continue

        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load TMF data from {path}: {e}")

    return {"results": {}}


# === Type56 Output Cache ===

def _is_type56_id(tmf_id: str) -> bool:
    """Check if a TMF component_id refers to Type56."""
    return tmf_id.lower() in ("type56", "type 56")


def _resolve_type56_building_path(
    source_file: str,
    bldg_rel: str,
) -> Optional[str]:
    """Resolve a Type56 building file path across Windows host and Docker layouts.

    Training data stores ``source_file`` as a Windows absolute path (e.g.
    ``d:\\TRNSYS18_03\\Examples\\...\\Building_imported.dck``). The old logic used
    ``os.path.dirname`` / ``os.path.join`` to derive the building file path, which
    silently breaks inside the Linux Docker container: ``posixpath`` does not
    recognize backslashes, so ``dirname`` returns an empty string and the building
    file is never found (every Type56 then falls back to "Output N").

    This helper splits ``source_file`` on BOTH separators (platform-independent),
    rebuilds the building file path, and returns the first candidate that exists:

    1. Original joined path (local Windows with TRNSYS18_03 installed).
    2. TRNSYS18_03 -> TRNSYS18_04 substituted path (local Windows, this project).
    3. Docker mount path ``/mnt/trnsys/TRNSYS18_04/...`` via the shared
       ``tmf_tools._normalize_type56_building_file_path`` mapping (container).

    Args:
        source_file: DCK original absolute path from training data.
        bldg_rel: Building file path from the Type56 unit's external_files entry
            (usually relative, e.g. "Building.b18").

    Returns:
        The first existing absolute path, or None if none resolve.
    """
    if not source_file or not bldg_rel:
        return None

    # Platform-independent join: never rely on os.path for Windows-style inputs.
    sf = str(source_file).strip().strip('"').replace("\\", "/")
    rel = str(bldg_rel).strip().strip('"').replace("\\", "/")
    dck_dir = sf.rsplit("/", 1)[0] if "/" in sf else ""
    if not dck_dir:
        return None
    joined = f"{dck_dir}/{rel}"

    candidates: List[Optional[str]] = [
        joined,
        joined.replace("TRNSYS18_03", "TRNSYS18_04"),
    ]

    # Reuse the shared Docker-mount mapping from tmf_tools (maps D:/TRNSYS18_0x ->
    # /mnt/trnsys/TRNSYS18_04/...). Imported lazily to avoid any import-order issues.
    try:
        from .tmf_tools import _normalize_type56_building_file_path

        candidates.append(_normalize_type56_building_file_path(joined))
    except ImportError:
        logger.debug("Could not import _normalize_type56_building_file_path from tmf_tools")

    seen = set()
    for cand in candidates:
        if not cand or cand in seen:
            continue
        seen.add(cand)
        if os.path.exists(cand):
            return cand

    return None


@lru_cache(maxsize=1)
def _build_type56_output_cache() -> Dict[str, Dict[int, List[Dict[str, Any]]]]:
    """Build a cache of Type56 output port names for each training DCK file.

    Parses the .b18/.b17/.bui building description file for each DCK that
    contains a Type56 component, and caches the resolved output list.

    Returns:
        Dict mapping dck_filename -> {type56_unit_number: [output_dicts]}.
        Each output_dict has keys: index, name, unit, description.
        Returns empty dict on failure.
    """
    dck_data = _load_training_dck_data()
    cache: Dict[str, Dict[int, List[Dict[str, Any]]]] = {}

    # Import the Type56 building file parser from tmf_tools
    try:
        from .tmf_tools import _get_type56_spec_from_building_file
    except ImportError:
        logger.warning("Could not import _get_type56_spec_from_building_file from tmf_tools")
        return cache

    for dck_result in dck_data.get("results", []):
        filename = dck_result.get("filename", "")
        source_file = dck_result.get("source_file", "")

        for u in dck_result.get("units", []):
            if u.get("type_number") != "56":
                continue

            unit_num = u.get("unit_number")
            ext_files = u.get("external_files", [])
            if not ext_files:
                continue

            # Get building file relative path
            bldg_rel = ext_files[0].get("file_path", "")
            if not bldg_rel:
                continue

            # Resolve building file path across Windows host and Docker layouts.
            # (Old os.path.dirname/join logic broke in the Linux container because
            # posixpath ignores the backslashes in the Windows-style source_file.)
            abs_path = _resolve_type56_building_path(source_file, bldg_rel)

            if not abs_path:
                logger.debug(
                    f"Type56 building file not found for {filename}: {bldg_rel} "
                    f"(source_file: {source_file})"
                )
                continue

            # Parse building file to get outputs
            try:
                spec = _get_type56_spec_from_building_file(abs_path)
                outputs = spec.get("outputs", [])
                if outputs:
                    if filename not in cache:
                        cache[filename] = {}
                    cache[filename][unit_num] = outputs
                    logger.debug(
                        f"Cached {len(outputs)} Type56 outputs for {filename} "
                        f"(unit {unit_num}) from {os.path.basename(abs_path)}"
                    )
            except Exception as e:
                logger.debug(f"Failed to parse Type56 building file for {filename}: {e}")

    logger.info(f"Type56 output cache built: {len(cache)} DCK files with resolved outputs")
    return cache


# === Helper Functions ===

def _extract_tmf_id_from_model_path(model_path: str) -> str:
    """Extract the TMF component id from Windows or POSIX-style model_path."""
    clean_path = str(model_path).strip().strip('"').strip("'")
    basename = re.split(r"[\\/]", clean_path)[-1]
    return re.sub(r"\.tmf$", "", basename, flags=re.IGNORECASE)


def _get_tmf_component_id(unit: Dict[str, Any]) -> str:
    """Extract TMF component_id from a unit's model_path.

    For UNIT type: extracts the .tmf filename prefix from model_path.
    For EQUATIONS type: returns "EQUATIONS".

    Args:
        unit: A unit dict from parsed DCK data.

    Returns:
        TMF component_id (e.g., "Type15-6", "Type65d") or "EQUATIONS".
    """
    unit_type = unit.get("unit_type", "")
    if unit_type == "EQUATIONS":
        return "EQUATIONS"

    model_path = unit.get("metadata", {}).get("model_path", "")
    if model_path:
        tmf_id = _extract_tmf_id_from_model_path(model_path)
        if tmf_id:
            return tmf_id

    # Fallback to full_type_id
    return unit.get("full_type_id", f"Type{unit.get('type_number', '?')}")


def _build_unit_map(dck_result: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    """Build unit_number -> unit info mapping for a single DCK file.

    Returns:
        Dict mapping unit_number to {tmf_id, unit_name, unit_type, equations}.
    """
    unit_map = {}
    for u in dck_result.get("units", []):
        unit_num = u.get("unit_number")
        if unit_num is None:
            continue
        tmf_id = _get_tmf_component_id(u)
        unit_name = u.get("metadata", {}).get("unit_name", "")
        unit_map[unit_num] = {
            "tmf_id": tmf_id,
            "unit_name": unit_name,
            "unit_type": u.get("unit_type", ""),
            "equations": u.get("equations", []),
        }
    return unit_map


def _normalize_type_id(type_id: str) -> str:
    """Normalize a type_id for matching (case-insensitive, with Type prefix)."""
    tid = type_id.strip()
    if not tid.lower().startswith("type") and tid != "EQUATIONS":
        tid = f"Type{tid}"
    return tid


def _tmf_id_matches(tmf_id: str, target_type: str) -> bool:
    """Check if a TMF component_id matches the target type.

    Supports both exact match and base-type match:
    - Exact: "Type15-6" matches "Type15-6"
    - Base: "Type15-6" matches "Type15" (base type number)
    - EQUATIONS: "EQUATIONS" matches "EQUATIONS"
    """
    if tmf_id == target_type:
        return True

    # Base type number match (e.g., "Type15-6" matches "Type15")
    base_match_tmf = re.match(r"(Type\d+)", tmf_id, re.IGNORECASE)
    base_match_target = re.match(r"(Type\d+)", target_type, re.IGNORECASE)
    if base_match_tmf and base_match_target:
        if base_match_tmf.group(1).lower() == base_match_target.group(1).lower():
            return True

    return False


def _resolve_output_name(
    tmf_id: str,
    output_index: int,
    tmf_results: Dict[str, Any],
    type56_outputs: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Resolve an output port number to its name using TMF data.

    For Type56 components, uses pre-parsed building file outputs instead of
    TMF data (since Type56 TMF only has a placeholder output).

    Args:
        tmf_id: TMF component_id (e.g., "Type15-6").
        output_index: 1-based output index.
        tmf_results: TMF parsed results dict.
        type56_outputs: Optional pre-parsed Type56 output list from building file.
            Each dict has keys: index, name, unit, description.

    Returns:
        Output name with unit, e.g., "Dry bulb temperature [C]".
        Falls back to "Output {index}" if not found.
    """
    # Type56 special handling: use building-file-parsed outputs
    if _is_type56_id(tmf_id) and type56_outputs:
        for o in type56_outputs:
            if o.get("index") == output_index:
                name = o.get("name", f"Output {output_index}")
                unit = o.get("unit", "")
                if unit:
                    return f"{name} [{unit}]"
                return name
        # If output_index exceeds parsed outputs, fall through to generic
        return f"Output {output_index}"

    tmf_key = f"{tmf_id}.tmf"
    tmf_entry = tmf_results.get(tmf_key, {})
    outputs = tmf_entry.get("outputs", [])

    for o in outputs:
        if o.get("index") == output_index:
            name = o.get("name", f"Output {output_index}")
            unit = o.get("unit", "")
            if unit:
                return f"{name} [{unit}]"
            return name

    return f"Output {output_index}"


def _resolve_input_name(
    tmf_id: str,
    input_index: int,
    tmf_results: Dict[str, Any],
    dck_unit: Optional[Dict[str, Any]] = None,
    dck_inputs: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Resolve an input port number to its name.

    For Type56: prioritizes DCK raw_line comments over TMF data. Type56's TMF
    input list only contains a single placeholder input, because Type56 inputs
    are dynamically defined by the building file; the DCK raw_line comment
    carries the real per-port name. TMF is used only as a last-resort fallback.

    For all other components: prioritizes DCK raw_line comments over TMF data,
    because many components have flexible inputs where the actual port names
    in DCK files differ from the TMF template (e.g., "Zone temperature-1"
    instead of "Zone temperature", or expanded ports beyond TMF count).

    Args:
        tmf_id: TMF component_id.
        input_index: 0-based input index (as stored in DCK connections).
        tmf_results: TMF parsed results dict.
        dck_unit: Optional unit dict from DCK for raw_line fallback.
        dck_inputs: Optional inputs list from DCK unit for raw_line fallback.

    Returns:
        Input name, e.g., "Zone temperature-1" or "Ambient temperature [C]".
    """
    # Type56 special case: DCK raw_line FIRST, then TMF fallback.
    # Type56's TMF input list contains only a single placeholder input
    # ('Inputs - to update attach files in "External Files" tab' at index 1),
    # because Type56 inputs are dynamically defined by the building file.
    # The previous "TMF first" order incorrectly matched port 1
    # (input_index=0 -> tmf_input_index=1) to that placeholder, while ports 2+
    # had no TMF match and correctly fell through to raw_line. The DCK raw_line
    # comment carries the real per-port name for EVERY port (e.g.
    # "Weather:Dry bulb temperature -> 1- TAMB"), so prioritizing it fixes
    # port 1 without changing the resolution of ports 2+.
    if _is_type56_id(tmf_id):
        if dck_inputs:
            for dck_inp in dck_inputs:
                if dck_inp.get("index") == input_index:
                    raw_line = dck_inp.get("raw_line", "")
                    comment_match = re.search(r"!\s*(.+)", raw_line)
                    if comment_match:
                        return comment_match.group(1).strip()
        # Fallback to TMF only when no raw_line comment is available
        tmf_input_index = input_index + 1
        tmf_key = f"{tmf_id}.tmf"
        tmf_entry = tmf_results.get(tmf_key, {})
        inputs = tmf_entry.get("inputs", [])
        for inp in inputs:
            if inp.get("index") == tmf_input_index:
                name = inp.get("name", "")
                if name:
                    unit = inp.get("unit", "")
                    if unit:
                        return f"{name} [{unit}]"
                    return name
        return f"Input {input_index + 1}"

    # For all other components: DCK raw_line FIRST (actual port names from
    # training data are more accurate than TMF templates for flexible-input
    # components where port indices shift based on parameter configuration)
    if dck_inputs:
        for dck_inp in dck_inputs:
            if dck_inp.get("index") == input_index:
                raw_line = dck_inp.get("raw_line", "")
                comment_match = re.search(r"!\s*(.+)", raw_line)
                if comment_match:
                    comment = comment_match.group(1).strip()
                    # Extract target input name from "SourceName:OutputName ->TargetInputName"
                    arrow_match = re.search(r"->\s*(.+)", comment)
                    if arrow_match:
                        return arrow_match.group(1).strip()
                    # Handle "[unconnected] InputName"
                    unconn_match = re.search(r"\[unconnected\]\s*(.+)", comment)
                    if unconn_match:
                        return unconn_match.group(1).strip()
                    # Return full comment if no pattern matched
                    return comment

    # Fallback: TMF data
    tmf_input_index = input_index + 1
    tmf_key = f"{tmf_id}.tmf"
    tmf_entry = tmf_results.get(tmf_key, {})
    inputs = tmf_entry.get("inputs", [])
    for inp in inputs:
        if inp.get("index") == tmf_input_index:
            name = inp.get("name", "")
            if name:
                unit = inp.get("unit", "")
                if unit:
                    return f"{name} [{unit}]"
                return name

    return f"Input {input_index + 1}"


def _resolve_equation_variable(
    unit_map: Dict[int, Dict[str, Any]],
    source_variable: str,
) -> Tuple[str, str]:
    """Find which equation block defines a variable.

    Returns:
        Tuple of (equation_unit_name, variable_name).
    """
    for unit_num, info in unit_map.items():
        if info["unit_type"] == "EQUATIONS":
            for eq in info.get("equations", []):
                if eq.get("variable") == source_variable:
                    return info["unit_name"], source_variable
    return "Unknown_Equation", source_variable


# === Core Analysis Functions ===

def _analyze_connections_for_pair(
    source_type: str,
    target_type: str,
) -> List[Dict[str, Any]]:
    """Analyze all connections between two component types across training data.

    Returns a list of resolved connection records with port names.
    For Type56 source components, uses building-file-parsed outputs for accurate
    port name resolution instead of the TMF placeholder.
    """
    dck_data = _load_training_dck_data()
    tmf_data = _load_tmf_data()
    tmf_results = tmf_data.get("results", {})

    source_type_norm = _normalize_type_id(source_type)
    target_type_norm = _normalize_type_id(target_type)

    # Pre-load Type56 output cache if source type is Type56
    type56_cache = None
    if _is_type56_id(source_type_norm):
        type56_cache = _build_type56_output_cache()

    all_resolved = []

    for dck_result in dck_data.get("results", []):
        filename = dck_result.get("filename", "unknown")
        unit_map = _build_unit_map(dck_result)
        connections = dck_result.get("connections", [])

        # Build reverse lookup: find units matching source and target types
        source_units = {}
        target_units = {}
        for unit_num, info in unit_map.items():
            if _tmf_id_matches(info["tmf_id"], source_type_norm):
                source_units[unit_num] = info
            if _tmf_id_matches(info["tmf_id"], target_type_norm):
                target_units[unit_num] = info

        if not source_units or not target_units:
            continue

        # Find DCK unit inputs for target units (for raw_line fallback)
        target_dck_inputs = {}
        for u in dck_result.get("units", []):
            if u.get("unit_number") in target_units:
                target_dck_inputs[u["unit_number"]] = u.get("inputs", [])

        # Get Type56 outputs for this DCK file (if applicable)
        dck_type56_outputs = {}
        if type56_cache and filename in type56_cache:
            dck_type56_outputs = type56_cache[filename]

        # Process connections
        for conn in connections:
            conn_type = conn.get("connection_type", "")

            if conn_type == "unit_to_unit":
                src_unit = conn.get("source_unit")
                tgt_unit = conn.get("target_unit")

                if src_unit in source_units and tgt_unit in target_units:
                    src_info = source_units[src_unit]
                    tgt_info = target_units[tgt_unit]
                    src_output = conn.get("source_output", 0)
                    tgt_input = conn.get("target_input", 0)

                    # Get Type56 outputs for this specific source unit
                    t56_outputs = dck_type56_outputs.get(src_unit) if dck_type56_outputs else None

                    src_output_name = _resolve_output_name(
                        src_info["tmf_id"], src_output, tmf_results,
                        type56_outputs=t56_outputs,
                    )
                    tgt_input_name = _resolve_input_name(
                        tgt_info["tmf_id"],
                        tgt_input,
                        tmf_results,
                        dck_inputs=target_dck_inputs.get(tgt_unit),
                    )

                    all_resolved.append({
                        "filename": filename,
                        "connection_type": "component_to_component",
                        "source_tmf_id": src_info["tmf_id"],
                        "source_unit_name": src_info["unit_name"],
                        "source_output_index": src_output,
                        "source_output_name": src_output_name,
                        "target_tmf_id": tgt_info["tmf_id"],
                        "target_unit_name": tgt_info["unit_name"],
                        "target_input_index": tgt_input,
                        "target_input_name": tgt_input_name,
                    })

            elif conn_type == "equation_to_unit":
                # V3.4: Renamed from variable_to_unit, added source_equation_unit field
                tgt_unit = conn.get("target_unit")
                src_var = conn.get("source_variable", "")
                src_eq_unit = conn.get("source_equation_unit")

                if tgt_unit in target_units:
                    tgt_info = target_units[tgt_unit]
                    tgt_input = conn.get("target_input", 0)

                    # Check if source equation unit matches source_type
                    if src_eq_unit in source_units:
                        src_info = source_units[src_eq_unit]

                        tgt_input_name = _resolve_input_name(
                            tgt_info["tmf_id"],
                            tgt_input,
                            tmf_results,
                            dck_inputs=target_dck_inputs.get(tgt_unit),
                        )
                        all_resolved.append({
                            "filename": filename,
                            "connection_type": "equation_to_component",
                            "source_tmf_id": "EQUATIONS",
                            "source_unit_name": src_info["unit_name"],
                            "source_equation_unit": src_eq_unit,
                            "source_output_index": 0,
                            "source_output_name": f"Variable: {src_var}",
                            "target_tmf_id": tgt_info["tmf_id"],
                            "target_unit_name": tgt_info["unit_name"],
                            "target_input_index": tgt_input,
                            "target_input_name": tgt_input_name,
                        })

            elif conn_type == "unit_to_equation":
                # V3.4: New connection type (component → equation)
                src_unit = conn.get("source_unit")
                tgt_eq_unit = conn.get("target_equation_unit")

                if src_unit in source_units and tgt_eq_unit in target_units:
                    src_info = source_units[src_unit]
                    tgt_info = target_units[tgt_eq_unit]
                    src_output = conn.get("source_output", 0)
                    tgt_input = conn.get("target_input", 0)

                    # Get Type56 outputs for this specific source unit
                    t56_outputs = dck_type56_outputs.get(src_unit) if dck_type56_outputs else None

                    src_output_name = _resolve_output_name(
                        src_info["tmf_id"], src_output, tmf_results,
                        type56_outputs=t56_outputs,
                    )

                    all_resolved.append({
                        "filename": filename,
                        "connection_type": "component_to_equation",
                        "source_tmf_id": src_info["tmf_id"],
                        "source_unit_name": src_info["unit_name"],
                        "source_output_index": src_output,
                        "source_output_name": src_output_name,
                        "target_tmf_id": "EQUATIONS",
                        "target_unit_name": tgt_info["unit_name"],
                        "target_equation_unit": tgt_eq_unit,
                        "target_input_index": tgt_input,
                        "target_input_name": f"Equation input {tgt_input}",
                    })

            elif conn_type == "equation_to_equation":
                # V3.4: New connection type (equation → equation)
                src_eq_unit = conn.get("source_equation_unit")
                tgt_eq_unit = conn.get("target_equation_unit")
                src_var = conn.get("source_variable", "")
                tgt_var = conn.get("target_variable", "")

                if src_eq_unit in source_units and tgt_eq_unit in target_units:
                    src_info = source_units[src_eq_unit]
                    tgt_info = target_units[tgt_eq_unit]

                    all_resolved.append({
                        "filename": filename,
                        "connection_type": "equation_to_equation",
                        "source_tmf_id": "EQUATIONS",
                        "source_unit_name": src_info["unit_name"],
                        "source_equation_unit": src_eq_unit,
                        "source_output_index": 0,
                        "source_output_name": f"Variable: {src_var}",
                        "target_tmf_id": "EQUATIONS",
                        "target_unit_name": tgt_info["unit_name"],
                        "target_equation_unit": tgt_eq_unit,
                        "target_input_index": 0,
                        "target_input_name": f"Variable: {tgt_var}",
                    })

    return all_resolved


def _aggregate_patterns(
    resolved_connections: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Aggregate resolved connections into frequency-ranked patterns.

    Groups by (source_output_name, target_input_name) and counts frequency.
    """
    pattern_counter = Counter()
    pattern_details = {}

    for conn in resolved_connections:
        key = (
            conn["source_output_name"],
            conn["target_input_name"],
            conn["source_output_index"],
            conn["target_input_index"],
        )
        pattern_counter[key] += 1

        if key not in pattern_details:
            pattern_details[key] = {
                "source_tmf_id": conn["source_tmf_id"],
                "target_tmf_id": conn["target_tmf_id"],
                "source_output_name": conn["source_output_name"],
                "source_output_index": conn["source_output_index"],
                "target_input_name": conn["target_input_name"],
                "target_input_index": conn["target_input_index"],
                "connection_type": conn["connection_type"],
                "example_files": [],
            }
        if len(pattern_details[key]["example_files"]) < 3:
            fn = conn["filename"]
            if fn not in pattern_details[key]["example_files"]:
                pattern_details[key]["example_files"].append(fn)

    # Sort by frequency descending
    sorted_patterns = []
    for key, count in pattern_counter.most_common():
        detail = pattern_details[key]
        detail["frequency"] = count
        sorted_patterns.append(detail)

    return sorted_patterns


# === Tool Functions ===

@tool
def search_connection_patterns(
    source_type: str,
    target_type: str,
    limit: int = 50,
) -> str:
    """Search for connection patterns between two component types with PORT NAMES.

    Searches the training DCK dataset (196 files, 10000+ connections) and returns
    connection patterns with RESOLVED PORT NAMES from TMF specifications, not just
    port numbers. This ensures physically correct connections.

    CRITICAL: Use exact TMF component IDs (e.g., "Type15-6" not "Type15",
    "Type65d" not "Type65"). For equation blocks, use "EQUATIONS".

    Args:
        source_type: Source component TMF ID (e.g., "Type15-6", "Type56", "EQUATIONS").
        target_type: Target component TMF ID (e.g., "Type56", "Type65d").
        limit: Maximum number of patterns to return (default: 50).

    Returns:
        Formatted string with connection patterns including port names, units,
        and frequency statistics. Each pattern shows:
        - Source output NAME and unit (not just port number)
        - Target input NAME and unit (not just port number)
        - Frequency across training DCK files
        - Port numbers as reference (in parentheses)
    """
    try:
        resolved = _analyze_connections_for_pair(source_type, target_type)

        if not resolved:
            return (
                f"No connection patterns found between {source_type} and {target_type}. "
                f"Check component IDs or try base type (e.g., 'Type15' instead of 'Type15-6')."
            )

        patterns = _aggregate_patterns(resolved)
        total_instances = sum(p["frequency"] for p in patterns)
        unique_files = len(set(c["filename"] for c in resolved))

        lines = [
            f"=== Connection Patterns: {source_type} -> {target_type} ===",
            f"Found in {unique_files} DCK files, {total_instances} connection instances",
            f"Showing {min(limit, len(patterns))} of {len(patterns)} unique patterns:",
            "",
        ]

        for i, p in enumerate(patterns[:limit], 1):
            freq_pct = round(p["frequency"] / unique_files * 100, 1) if unique_files > 0 else 0
            lines.append(
                f"Pattern {i} (frequency: {p['frequency']}/{unique_files} = {freq_pct}%):"
            )
            lines.append(
                f"  {p['source_tmf_id']} output \"{p['source_output_name']}\" (port {p['source_output_index']})"
                f" -> {p['target_tmf_id']} input \"{p['target_input_name']}\" (port {p['target_input_index'] + 1})"
            )
            if p["example_files"]:
                lines.append(f"  Example DCK files: {', '.join(p['example_files'])}")
            lines.append("")

        lines.append("NOTE: Match connections by PORT NAME and physical quantity, not just port number.")
        lines.append("Port numbers may differ for flexible-input components (Type56, Type65d, etc.).")
        lines.append("")
        lines.append(
            "[!] CRITICAL — These patterns are AGGREGATED ACROSS DIFFERENT DCK files "
            "(and different building files for Type56), then ranked by frequency. They are "
            "NOT a single self-consistent wiring map. The SAME port number can mean DIFFERENT "
            "physical quantities in different files (e.g. Type56 input port 7 may be ground "
            "reflectance in one file but tilted-surface radiation in another). DO NOT wire by "
            "port number from this aggregated list. Use it ONLY to (a) see the mainstream "
            "consensus and (b) pick a candidate reference DCK from 'Example DCK files'. For the "
            "ACTUAL connections of a flexible-I/O component (Type56, Type65d, Type534, etc.), "
            "call get_connection_pattern(dck_filename=...) on ONE reference DCK to get a "
            "self-consistent map, then match every connection by PORT NAME and physical quantity."
        )

        return "\n".join(lines)

    except Exception as e:
        logger.error(f"Error in search_connection_patterns: {e}")
        return f"Error searching connection patterns: {str(e)}"


@tool
def get_connection_pattern(
    source_type: str,
    target_type: str,
    dck_filename: Optional[str] = None,
) -> str:
    """Get detailed connection pattern between two component types with PORT NAMES.

    Returns the complete set of connections between source and target types,
    with port names resolved from TMF specifications. If dck_filename is specified,
    returns connections from that specific DCK file; otherwise returns the most
    common pattern across all training data.

    CRITICAL: Use exact TMF component IDs (e.g., "Type15-6" not "Type15").

    Args:
        source_type: Source component TMF ID (e.g., "Type15-6", "EQUATIONS").
        target_type: Target component TMF ID (e.g., "Type56", "Type65d").
        dck_filename: Optional specific DCK filename to query (e.g., "Building.dck").

    Returns:
        Detailed connection mapping with port names, units, and port numbers.
    """
    try:
        resolved = _analyze_connections_for_pair(source_type, target_type)

        if not resolved:
            return (
                f"No connection patterns found between {source_type} and {target_type}."
            )

        if dck_filename:
            # Filter to specific DCK file
            filtered = [c for c in resolved if c["filename"] == dck_filename]
            if not filtered:
                available = sorted(set(c["filename"] for c in resolved))[:10]
                return (
                    f"No connections found in '{dck_filename}' for {source_type} -> {target_type}. "
                    f"Available DCK files with this pattern: {', '.join(available)}"
                )
            resolved = filtered

        # Group by filename to show per-file patterns
        by_file = defaultdict(list)
        for conn in resolved:
            by_file[conn["filename"]].append(conn)

        lines = [
            f"=== Detailed Connection Pattern: {source_type} -> {target_type} ===",
        ]

        if dck_filename:
            lines.append(f"From DCK file: {dck_filename}")
            lines.append("")
            for conn in resolved:
                lines.append(
                    f"  {conn['source_tmf_id']} output \"{conn['source_output_name']}\" (port {conn['source_output_index']})"
                    f" -> {conn['target_tmf_id']} input \"{conn['target_input_name']}\" (port {conn['target_input_index'] + 1})"
                )
        else:
            # Show the most representative file (most connections)
            best_file = max(by_file.keys(), key=lambda f: len(by_file[f]))
            best_conns = by_file[best_file]
            total_files = len(by_file)

            lines.append(
                f"Most representative pattern from '{best_file}' "
                f"(found in {total_files} DCK files total):"
            )
            lines.append("")

            # Sort by target input index for readability
            best_conns.sort(key=lambda c: c["target_input_index"])

            for conn in best_conns:
                lines.append(
                    f"  {conn['source_tmf_id']} output \"{conn['source_output_name']}\" (port {conn['source_output_index']})"
                    f" -> {conn['target_tmf_id']} input \"{conn['target_input_name']}\" (port {conn['target_input_index'] + 1})"
                )

            # Also show aggregated patterns
            patterns = _aggregate_patterns(resolved)
            lines.append("")
            lines.append(f"--- Aggregated across {total_files} DCK files ---")
            for p in patterns[:30]:
                lines.append(
                    f"  [{p['frequency']}x] \"{p['source_output_name']}\" (port {p['source_output_index']})"
                    f" -> \"{p['target_input_name']}\" (port {p['target_input_index'] + 1})"
                )

        lines.append("")
        lines.append("IMPORTANT: Match by PORT NAME, not port number. Port numbers may vary for flexible components.")
        lines.append("")
        lines.append(
            "[!] PORT-ORDERING IS AUTHORITATIVE HERE — For a flexible-I/O target "
            "(Type56, Type151, Type152, Type1255, Type65d, ...), the port SEQUENCE shown "
            "above is the real, self-consistent ordering used by this reference DCK. "
            "NOTE: an 'instance' is whatever the controlling parameter counts — a zone, tank "
            "node, air node, inlet/outlet port, comparator, floor, control, heating/cooling "
            "stage, load, level, etc. — NOT necessarily a zone. "
            "Read off HOW the multi-instance template inputs are arranged: either BLOCKED "
            "by physical quantity (all instances' quantity-1 first, then all instances' "
            "quantity-2, ... — e.g. Type151/Type152, whose instances are zones: all zones' "
            "temperature, then all zones' humidity, then all zones' load) or GROUPED by "
            "instance (instance-1's full variable set, then instance-2's full set, ... — "
            "e.g. Type1255 grouped per zone, Type649 grouped per inlet). This ordering "
            "DIFFERS per component type — do NOT rebuild the sequence from a base I/O "
            "template by assuming the template simply repeats per instance."
        )
        lines.append(
            "[!] IF INSTANCE COUNT DIFFERS — This reference DCK may have a different number "
            "of instances (zones / tank nodes / inlets / stages / ... whatever this "
            "component's controlling parameter counts) than your modeling task (e.g. the "
            "reference has 5 instances but your task needs 10). The reference gives you the "
            "ORDERING PATTERN (blocked vs grouped) and the per-instance port NAMES, NOT the "
            "final port numbers. Re-expand that SAME pattern for your task's actual instance "
            "count N, then match every connection by PORT NAME + physical quantity. The "
            "reference is guidance for the layout pattern, not a guaranteed 1:1 port-number copy."
        )

        return "\n".join(lines)

    except Exception as e:
        logger.error(f"Error in get_connection_pattern: {e}")
        return f"Error getting connection pattern: {str(e)}"


@tool
def find_dcks_with_components(
    component_types: List[str],
    limit: int = 10,
) -> str:
    """Find training DCK files that contain ALL specified component types.

    Searches the training dataset to find DCK files containing all the
    specified component types. Useful for finding reference systems with
    similar component combinations.

    CRITICAL: Use exact TMF component IDs (e.g., "Type15-6" not "Type15").
    For equation blocks, use "EQUATIONS".

    Args:
        component_types: List of TMF component IDs to search for.
            Example: ["Type15-6", "Type56", "Type65d"]
        limit: Maximum number of DCK files to return (default: 10).

    Returns:
        List of matching DCK files with their component inventories.
    """
    try:
        dck_data = _load_training_dck_data()
        normalized_types = [_normalize_type_id(t) for t in component_types]

        matches = []

        for dck_result in dck_data.get("results", []):
            filename = dck_result.get("filename", "unknown")
            unit_map = _build_unit_map(dck_result)

            # Collect all TMF IDs in this DCK
            dck_tmf_ids = set()
            for info in unit_map.values():
                dck_tmf_ids.add(info["tmf_id"])

            # Check if all requested types are present
            all_found = True
            for req_type in normalized_types:
                found = any(
                    _tmf_id_matches(tmf_id, req_type)
                    for tmf_id in dck_tmf_ids
                )
                if not found:
                    all_found = False
                    break

            if all_found:
                # Build component summary
                comp_summary = []
                for unit_num, info in sorted(unit_map.items()):
                    comp_summary.append(
                        f"Unit {unit_num}: {info['tmf_id']} ({info['unit_name']})"
                    )

                conn_count = len(dck_result.get("connections", []))
                matches.append({
                    "filename": filename,
                    "components": comp_summary,
                    "connection_count": conn_count,
                })

            if len(matches) >= limit:
                break

        if not matches:
            type_str = ", ".join(component_types)
            return (
                f"No DCK files found containing all of: {type_str}. "
                f"Try using base type numbers (e.g., 'Type15' instead of 'Type15-6') "
                f"or reduce the component list."
            )

        type_str = ", ".join(component_types)
        lines = [
            f"=== DCK Files Containing: {type_str} ===",
            f"Found {len(matches)} matching DCK files (showing up to {limit}):",
            "",
        ]

        for i, m in enumerate(matches, 1):
            lines.append(f"--- {i}. {m['filename']} ({m['connection_count']} connections) ---")
            for comp in m["components"]:
                lines.append(f"  {comp}")
            lines.append("")

        return "\n".join(lines)

    except Exception as e:
        logger.error(f"Error in find_dcks_with_components: {e}")
        return f"Error finding DCK files: {str(e)}"


# === Connection Unit / Dimension Verification ===
# Maps a normalized unit string to a coarse physical DIMENSION class.
# Derived from a scan of all 765 TMF files' real unit strings, so every unit
# below actually occurs in the component library.
_UNIT_DIMENSION_MAP = {
    # Temperature (absolute) and temperature difference
    "C": "TEMPERATURE", "K": "TEMPERATURE", "F": "TEMPERATURE", "deltaC": "TEMPERATURE",
    # Mass flow rate
    "kg/hr": "MASS_FLOW", "kg/s": "MASS_FLOW",
    # Volume flow rate
    "m^3/hr": "VOLUME_FLOW", "m^3/s": "VOLUME_FLOW", "l/hr": "VOLUME_FLOW",
    "l/s": "VOLUME_FLOW", "gal/hr": "VOLUME_FLOW", "Nm3/h": "VOLUME_FLOW", "Nm^3": "VOLUME_FLOW",
    # Power (energy rate)
    "kJ/hr": "POWER", "W": "POWER", "kW": "POWER",
    # Energy (accumulated)
    "kJ": "ENERGY", "J": "ENERGY", "Wh": "ENERGY", "kWh": "ENERGY",
    # Angle
    "degrees": "ANGLE",
    # Pressure
    "atm": "PRESSURE", "kPa": "PRESSURE", "bar": "PRESSURE", "BAR": "PRESSURE", "Pa": "PRESSURE",
    # Relative humidity (percentage based)
    "%": "REL_HUMIDITY", "% (base 100)": "REL_HUMIDITY", "% (base 1)": "REL_HUMIDITY",
    # Irradiance (area-normalized power)
    "kJ/hr.m^2": "IRRADIANCE", "W/m^2": "IRRADIANCE",
    # Electrical
    "V": "VOLTAGE", "A": "CURRENT", "amperes": "CURRENT", "mA": "CURRENT",
    # Velocity
    "m/s": "VELOCITY",
    # Length
    "m": "LENGTH", "mm": "LENGTH", "cm": "LENGTH", "km": "LENGTH",
    # Time
    "hr": "TIME", "day": "TIME", "s": "TIME",
    # Specific energy
    "kJ/kg": "SPECIFIC_ENERGY", "J/kg": "SPECIFIC_ENERGY",
    # Density
    "kg/m^3": "DENSITY",
    # Area
    "m^2": "AREA", "cm^2": "AREA",
}

# Wildcard / unspecified unit sentinels (NOT a real dimension). After
# _normalize_unit, a missing unit becomes "any"; "Fraction"/"ratio" are
# dimensionless wrappers that carry no physical dimension by themselves.
_WILDCARD_UNITS = {"any", "-", "", "fraction", "ratio", "frac"}

# Keyword-based dimension inference from PORT NAME, used when the unit is a
# wildcard/None. This is ESSENTIAL: humidity ratios and control signals carry
# NO unit in the TMF data, so a pure unit-string compare would silently miss a
# temperature->humidity-ratio mis-wire. Order matters: most specific first.
_NAME_DIMENSION_KEYWORDS = [
    ("relative humidity", "REL_HUMIDITY"),
    ("% relative", "REL_HUMIDITY"),
    ("humidity ratio", "HUMIDITY_RATIO"),
    ("absolute humidity", "HUMIDITY_RATIO"),
    ("moisture content", "HUMIDITY_RATIO"),
    ("control signal", "CONTROL"),
    ("control function", "CONTROL"),
    ("on/off", "CONTROL"),
    ("on-off", "CONTROL"),
    ("conditioning signal", "CONTROL"),
    ("coefficient of performance", "DIMENSIONLESS"),
    ("c.o.p", "DIMENSIONLESS"),
    ("part load ratio", "DIMENSIONLESS"),
    ("stoichiometric ratio", "DIMENSIONLESS"),
    ("bypass fraction", "DIMENSIONLESS"),
    ("fraction of", "DIMENSIONLESS"),
    ("temperature", "TEMPERATURE"),
    ("flow rate", "MASS_FLOW"),
    ("flowrate", "MASS_FLOW"),
    ("mass flow", "MASS_FLOW"),
    ("power", "POWER"),
    ("energy rate", "POWER"),
    ("heat transfer rate", "POWER"),
    ("pressure", "PRESSURE"),
    ("radiation", "IRRADIANCE"),
    ("irradiance", "IRRADIANCE"),
    ("voltage", "VOLTAGE"),
    ("current", "CURRENT"),
    ("velocity", "VELOCITY"),
    ("relative speed", "DIMENSIONLESS"),
]

# Built-in conversion instructions for same-dimension / unit-mismatch cases,
# keyed by (from_unit, to_unit) lower-cased. The value is a human-readable
# instruction; it is NOT auto-applied -- the LLM injects the Unit_Converter
# equation based on it.
_CONVERSION_FACTORS = {
    # Volume flow
    ("l/s", "l/hr"): "multiply by 3600",
    ("l/hr", "l/s"): "divide by 3600 (multiply by 0.000277778)",
    ("m^3/s", "m^3/hr"): "multiply by 3600",
    ("m^3/hr", "m^3/s"): "divide by 3600",
    # Mass flow
    ("kg/s", "kg/hr"): "multiply by 3600",
    ("kg/hr", "kg/s"): "divide by 3600 (multiply by 0.000277778)",
    # Power
    ("w", "kw"): "divide by 1000 (multiply by 0.001)",
    ("kw", "w"): "multiply by 1000",
    ("kj/hr", "kw"): "multiply by 0.000277778 (divide by 3600)",
    ("kw", "kj/hr"): "multiply by 3600",
    ("kj/hr", "w"): "divide by 3.6",
    ("w", "kj/hr"): "multiply by 3.6",
    # Energy
    ("j", "kj"): "divide by 1000",
    ("kj", "j"): "multiply by 1000",
    ("wh", "kj"): "multiply by 3.6",
    ("kj", "wh"): "divide by 3.6",
}

# Cross-dimension pairs that are physically convertible (not a wrong wiring):
# mass<->volume flow needs density rho; power<->energy is rate vs accumulated.
_COMPATIBLE_DIMENSION_PAIRS = {
    frozenset({"MASS_FLOW", "VOLUME_FLOW"}),
    frozenset({"POWER", "ENERGY"}),
}

# Flexible-I/O components -> append a port-ordering authority warning when any
# of them is a connection TARGET (their expanded port order is not uniform).
_VCU_FLEXIBLE_HINT_TYPES = {
    "Type49", "Type1244", "Type1255", "Type1256", "Type957",
    "Type158", "Type531", "Type533", "Type534", "Type541d", "Type550_IAM_Mode_2",
    "Type151", "Type152", "Type594", "Type595", "Type646", "Type647", "Type648",
    "Type649", "Type1260", "Type56",
    "Type24", "Type25b", "Type25c", "Type27c", "Type28b",
    "Type46a", "Type46b", "Type65a", "Type65c", "Type65d",
    "Type40", "Type150", "Type1502", "Type1503", "Type970", "Type971",
    "Type1250", "Type698", "Type660", "Type690", "Type759", "Type932", "Type2280",
}

_VCU_UNRESOLVED_DIMS = {"UNSPECIFIED", "UNKNOWN_UNIT"}

_VCU_FLEX_WARN_1 = (
    "[!] PORT-ORDERING IS AUTHORITATIVE HERE -- for a flexible-I/O target "
    "(Type56, Type151, Type152, Type1255, Type65d, ...) the real expanded port "
    "order is BLOCKED-by-quantity for some types (Type151/152: all zones' "
    "temperature, then all humidities, then all loads) and GROUPED-by-instance "
    "for others (Type1255 per zone, Type649 per inlet). Confirm the target port's "
    "true physical quantity via get_connection_pattern(dck_filename=...), not the "
    "single-instance base template."
)
_VCU_FLEX_WARN_2 = (
    "[!] IF INSTANCE COUNT DIFFERS -- the reference DCK's instance count (zones / "
    "tank nodes / inlets / stages / ... whatever the controlling parameter counts) "
    "may differ from this task; re-expand the SAME ordering pattern for the task's "
    "actual N and match every connection by PORT NAME + physical quantity, never by "
    "raw port number."
)


def _vcu_infer_dimension(unit_norm: str, port_name: str) -> str:
    """Infer a coarse physical dimension from a normalized unit, falling back
    to PORT NAME keywords when the unit is a wildcard/None."""
    u = (unit_norm or "").strip()
    u_low = u.lower()
    # 1) direct unit map (exact, then case-insensitive)
    if u in _UNIT_DIMENSION_MAP:
        return _UNIT_DIMENSION_MAP[u]
    for k, v in _UNIT_DIMENSION_MAP.items():
        if k.lower() == u_low:
            return v
    name_low = (port_name or "").lower()
    # 2) wildcard unit -> infer from port name
    if u_low in _WILDCARD_UNITS:
        for kw, dim in _NAME_DIMENSION_KEYWORDS:
            if kw in name_low:
                return dim
        return "UNSPECIFIED"
    # 3) unknown non-empty unit -> try port name, else flag unknown unit
    for kw, dim in _NAME_DIMENSION_KEYWORDS:
        if kw in name_low:
            return dim
    return "UNKNOWN_UNIT"


def _vcu_lookup_factor(src_unit_norm: str, tgt_unit_norm: str) -> Optional[str]:
    """Return a conversion instruction for a same-dimension unit mismatch, or
    None if no built-in factor is known."""
    a = (src_unit_norm or "").strip().lower()
    b = (tgt_unit_norm or "").strip().lower()
    if (a, b) in _CONVERSION_FACTORS:
        return _CONVERSION_FACTORS[(a, b)]
    if {a, b} == {"c", "k"}:
        return "add/subtract 273.15 (NOT a multiplicative factor)"
    return None


def _vcu_port_unit_and_name(
    type_id: str,
    port_kind: str,
    port_index: int,
    building_file_path: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Look up (unit_raw, name) for one component port via TMF.

    port_kind is 'output' or 'input'. Reuses tmf_tools so all Type56
    building-file / Docker-path handling is inherited. Returns
    (unit_raw, name, error_str); error_str is None on success.
    """
    try:
        from .tmf_tools import (
            _find_tmf_by_type,
            _normalize_type56_building_file_path,
        )
    except Exception as e:  # pragma: no cover - import guard
        return None, None, f"tmf_tools import failed: {e}"
    try:
        bfp = _normalize_type56_building_file_path(building_file_path)
        matches = _find_tmf_by_type(type_id, building_file_path=bfp)
    except Exception as e:
        return None, None, f"TMF lookup failed for {type_id}: {e}"
    if not matches:
        return None, None, f"no TMF spec for '{type_id}'"
    spec = matches[0]
    ports = spec.get("outputs", []) if port_kind == "output" else spec.get("inputs", [])
    for p in ports:
        if p.get("index") == port_index:
            return p.get("unit", ""), p.get("name", ""), None
    return None, None, (
        f"{type_id} {port_kind} port {port_index} not found in TMF "
        f"(template has {len(ports)} ports; for flexible components the real "
        f"port may be an expanded instance -- verify via get_connection_pattern)"
    )


def _vcu_classify_pair(pair: Dict[str, Any]) -> Dict[str, Any]:
    """Classify one already-decided connection pair into
    MATCH / CONVERT / MISMATCH / UNVERIFIED / ERROR."""
    from .tmf_tools import _normalize_unit

    cid = str(pair.get("id", "?"))

    # --- resolve SOURCE unit + name ---
    src_kind = pair.get("source_kind", "component")
    if src_kind == "equation":
        src_unit_raw = pair.get("source_unit", "")
        src_name = pair.get("source_var", "(equation var)")
        src_label = f'EQ:{src_name}'
        src_err = None
    else:
        src_unit_raw, src_name, src_err = _vcu_port_unit_and_name(
            str(pair.get("source_type", "")), "output",
            int(pair.get("source_output", -1)),
            pair.get("source_building_file"),
        )
        src_label = f'{pair.get("source_type","?")}:out{pair.get("source_output","?")} "{src_name}"'

    # --- resolve TARGET unit + name ---
    tgt_kind = pair.get("target_kind", "component")
    if tgt_kind == "equation":
        tgt_unit_raw = pair.get("target_unit", "")
        tgt_name = pair.get("target_var", "(equation var)")
        tgt_label = f'EQ:{tgt_name}'
        tgt_err = None
    else:
        tgt_unit_raw, tgt_name, tgt_err = _vcu_port_unit_and_name(
            str(pair.get("target_type", "")), "input",
            int(pair.get("target_input", -1)),
            pair.get("target_building_file"),
        )
        tgt_label = f'{pair.get("target_type","?")}:in{pair.get("target_input","?")} "{tgt_name}"'

    if src_err or tgt_err:
        return {
            "id": cid, "verdict": "ERROR",
            "line": f"[{cid}] {src_label} -> {tgt_label} | LOOKUP ERROR: "
                    f"{(src_err or '')} {(tgt_err or '')}".strip(),
        }

    src_norm = _normalize_unit(src_unit_raw)
    tgt_norm = _normalize_unit(tgt_unit_raw)
    src_dim = _vcu_infer_dimension(src_norm, src_name)
    tgt_dim = _vcu_infer_dimension(tgt_norm, tgt_name)

    base = (f"[{cid}] {src_label} [{src_norm}] (DIM={src_dim}) -> "
            f"{tgt_label} [{tgt_norm}] (DIM={tgt_dim})")

    # Unverifiable: either side has no resolvable dimension
    if src_dim in _VCU_UNRESOLVED_DIMS or tgt_dim in _VCU_UNRESOLVED_DIMS:
        return {"id": cid, "verdict": "UNVERIFIED",
                "line": base + " | UNVERIFIED (no resolvable dimension; manual check by port name)"}

    # Same dimension
    if src_dim == tgt_dim:
        if src_norm.lower() == tgt_norm.lower():
            return {"id": cid, "verdict": "MATCH", "line": base + " | MATCH"}
        factor = _vcu_lookup_factor(src_norm, tgt_norm)
        if factor:
            return {"id": cid, "verdict": "CONVERT",
                    "line": base + f" | UNIT MISMATCH -> CONVERT: {factor}"}
        return {"id": cid, "verdict": "CONVERT",
                "line": base + " | UNIT MISMATCH -> add Unit_Converter (factor unknown, derive it)"}

    # Different dimension but physically convertible
    if frozenset({src_dim, tgt_dim}) in _COMPATIBLE_DIMENSION_PAIRS:
        return {"id": cid, "verdict": "CONVERT",
                "line": base + " | CROSS-DIMENSION (convertible, e.g. needs density rho) -> add conversion equation"}

    # True dimension mismatch -> wrong connection
    return {"id": cid, "verdict": "MISMATCH",
            "line": base + " | DIMENSION MISMATCH -> WRONG CONNECTION, re-wire to a port of matching physical quantity"}


@tool
def verify_connection_units(connection_pairs: str) -> str:
    """Verify physical UNIT and DIMENSION consistency of ALREADY-DECIDED connections.

    Run this AFTER all connections are fixed (connection_generator Step 5) to
    catch the two catastrophic wiring faults:
      1. DIMENSION MISMATCH -- e.g. a temperature output wired into a humidity
         input. This is a WRONG connection that must be RE-WIRED, never masked.
      2. UNIT MISMATCH -- same dimension, different unit (e.g. L/s vs L/hr ->
         3600x error; kW vs W -> 1000x error). Fix by injecting a
         Unit_Converter_XXX equation (source * factor) between the two ports.

    It checks ALL 4 connection types (component->component, component->equation,
    equation->component, equation->equation) in ONE call, up to 10 pairs per
    call (split larger sets into batches). For COMPONENT ports it looks up the
    real unit + name from the TMF library (Type56 building-file handling
    included). For EQUATION ports it does NOT read any file -- you must pass the
    equation variable's unit, which you already know from that equation's
    `comment` field ("VAR [unit]; ...") in the state/equations you were given or
    just created. Do NOT confuse those state equations with training-set
    equations.

    Args:
        connection_pairs: A JSON array string (max 10 objects). Each object
            describes ONE decided connection. Fields:
              - id: short label for the pair (e.g. "C1").
              - source_kind: "component" or "equation".
              - For a component source: source_type (exact TMF id, e.g.
                "Type3b", "Type56"), source_output (1-based output port index),
                and optionally source_building_file (abs path to .b18/.b17/.bui
                for Type56).
              - For an equation source: source_var (variable name) and
                source_unit (its unit string, read from the equation comment;
                use "" or "-" for a dimensionless/control variable).
              - target_kind: "component" or "equation".
              - For a component target: target_type, target_input (1-based input
                port index), and optionally target_building_file (Type56).
              - For an equation target: target_var and target_unit.
            Example:
              [{"id":"C1","source_kind":"component","source_type":"Type3b",
                "source_output":3,"target_kind":"component","target_type":"Type56",
                "target_input":2,"target_building_file":"D:\\\\proj\\\\house.b18"},
               {"id":"C2","source_kind":"equation","source_var":"POWER_KW",
                "source_unit":"kW","target_kind":"component","target_type":"Type1234",
                "target_input":5}]

    Returns:
        A per-pair verdict report (MATCH / CONVERT / MISMATCH / UNVERIFIED /
        ERROR) plus a summary, recommended actions, and -- when a flexible-I/O
        component is a target -- a port-ordering authority warning.
    """
    try:
        try:
            pairs = json.loads(connection_pairs) if isinstance(connection_pairs, str) else connection_pairs
        except (TypeError, ValueError) as e:
            return f"ERROR: connection_pairs must be a JSON array string. Parse error: {e}"

        if not isinstance(pairs, list):
            return "ERROR: connection_pairs must be a JSON array of pair objects."
        if not pairs:
            return "ERROR: connection_pairs is empty; pass 1-10 decided connection pairs."
        if len(pairs) > 10:
            return (f"ERROR: too many pairs ({len(pairs)}). Max 10 per call to keep the "
                    f"check focused; split into batches of <=10.")

        results = [_vcu_classify_pair(p if isinstance(p, dict) else {}) for p in pairs]

        counts = {"MATCH": 0, "CONVERT": 0, "MISMATCH": 0, "UNVERIFIED": 0, "ERROR": 0}
        lines = ["=== Connection Unit & Dimension Verification ==="]
        for r in results:
            counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
            lines.append(r["line"])

        # Collect flexible-I/O component targets for the ordering warning.
        flex_types_seen = set()
        for p in pairs:
            if isinstance(p, dict) and p.get("target_kind", "component") == "component":
                t = p.get("target_type", "")
                if t in _VCU_FLEXIBLE_HINT_TYPES:
                    flex_types_seen.add(t)

        lines.append("")
        lines.append(
            f"SUMMARY: {counts['MATCH']} match, {counts['CONVERT']} convert, "
            f"{counts['MISMATCH']} MISMATCH(wrong), {counts['UNVERIFIED']} unverified, "
            f"{counts['ERROR']} error."
        )
        if counts["MISMATCH"]:
            lines.append(
                "ACTION (MISMATCH): different physical dimension means WRONG wiring "
                "(e.g. temperature->humidity). Re-wire to a source/target port of the "
                "SAME physical quantity. Do NOT mask a dimension mismatch with a "
                "conversion equation."
            )
        if counts["CONVERT"]:
            lines.append(
                "ACTION (CONVERT): same dimension, different unit. Inject a "
                "Unit_Converter_XXX equation (source * factor) and re-route the target "
                "input through it (source -> equation -> target)."
            )
        if counts["UNVERIFIED"]:
            lines.append(
                "ACTION (UNVERIFIED): no resolvable unit/dimension on at least one side "
                "(e.g. control signals, COP). Confirm by port NAME that the physical "
                "quantities match."
            )
        if flex_types_seen:
            lines.append("")
            lines.append(f"FLEXIBLE-I/O TARGETS PRESENT: {', '.join(sorted(flex_types_seen))}")
            lines.append(_VCU_FLEX_WARN_1)
            lines.append(_VCU_FLEX_WARN_2)

        return "\n".join(lines)

    except Exception as e:
        logger.error(f"Error in verify_connection_units: {e}")
        return f"Error verifying connection units: {str(e)}"


# === Tool Collection Functions ===

def get_connection_pattern_tools() -> List:
    """Get all connection pattern tools."""
    return [
        search_connection_patterns,
        get_connection_pattern,
        find_dcks_with_components,
        verify_connection_units,
    ]


def get_connection_pattern_tools_for_agent(agent_name: str) -> List:
    """Get connection pattern tools appropriate for a specific agent."""
    if agent_name in ["connection_generator", "validator", "supervisor"]:
        return get_connection_pattern_tools()
    # Equation generator only needs find_dcks_with_components to locate training
    # DCKs that contain specific component combinations (e.g., Type15 + Type56)
    # for bundle-equation pattern retrieval.
    if agent_name == "equation_generator":
        return [find_dcks_with_components]
    # Parameter configurator also benefits from find_dcks_with_components to
    # retrieve training DCKs containing a given component type, so that its
    # parameter configurations can serve as verified references.
    if agent_name == "parameter_configurator":
        return [find_dcks_with_components]
    return []


# === Utility Functions ===

def clear_connection_pattern_cache():
    """Clear the cached training data."""
    _select_training_dck_source.cache_clear()
    _load_training_dck_data.cache_clear()
    _load_tmf_data.cache_clear()
    _build_type56_output_cache.cache_clear()
    logger.info("Connection pattern cache cleared")
