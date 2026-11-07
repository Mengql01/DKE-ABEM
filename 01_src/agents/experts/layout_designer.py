"""
Layout Designer Expert Agent for TRNSYS DCK Generator.

Responsible for designing component layout and generating LINK_STYLE blocks.

Key Responsibilities:
1. Analyze connections and organize them by source component
2. Assign position coordinates to components based on data flow
3. Generate LINK_STYLE blocks with correct CONNECTION_SET format
4. Ensure layout is clear, readable and aesthetically pleasing

Design Principles:
- Layout based on data flow analysis (sources on left, sinks on right)
- LLM handles spatial design without external tool queries (tools disabled)
- Correct TRNSYS Studio LINK_STYLE format
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple, Set, Union

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from .base_expert import BaseExpert

logger = logging.getLogger(__name__)


def _snap_to_grid(value: int, grid: int = 20) -> int:
    """Snap a coordinate value to the nearest multiple of grid (default 20)."""
    return round(value / grid) * grid


def _fmt_coord(value) -> str:
    """Format a coordinate for DCK output.

    Drops a trailing ".0" so integral values render as integers (e.g. 315.0 ->
    "315") while keeping genuine fractional values intact (e.g. 326.8 -> "326.8").
    Variable icon sizes introduce half-pixel centres (width/2), so values may be
    int, float, or numeric strings.
    """
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if f == int(f):
        return str(int(f))
    return repr(f)


def _snap_to_y_grid(value: int) -> int:
    """Snap a Y-coordinate to the nearest valid y = 35 + 16n (n >= 0)."""
    n = round((value - 35) / 16)
    if n < 0:
        n = 0
    return 35 + 16 * n


# ======================================================================
# Icon Size Management - Support for variable component icon sizes
# ======================================================================

# Global cache for icon sizes
_ICON_SIZE_CACHE: Optional[Dict[str, Tuple[int, int]]] = None


def _load_icon_sizes() -> Dict[str, Tuple[int, int]]:
    """Load icon size lookup table from JSON file.

    Returns:
        Dictionary mapping component_id -> (width, height)
    """
    global _ICON_SIZE_CACHE

    if _ICON_SIZE_CACHE is not None:
        return _ICON_SIZE_CACHE

    try:
        # Get the original_trnsys_data directory (three levels up from experts/)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up: experts/ -> agents/ -> src/ -> trnsys_dck_generator/
        trnsys_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
        json_path = os.path.join(trnsys_root, "original_trnsys_data", "icon_sizes_lookup.json")

        with open(json_path, 'r', encoding='utf-8') as f:
            _ICON_SIZE_CACHE = json.load(f)
            logger.info(f"Loaded {len(_ICON_SIZE_CACHE)} icon sizes from {json_path}")
            return _ICON_SIZE_CACHE
    except Exception as e:
        logger.warning(f"Failed to load icon sizes: {e}. Using default 40x40 for all components.")
        _ICON_SIZE_CACHE = {}
        return _ICON_SIZE_CACHE


def extract_component_id(model_path: str) -> Optional[str]:
    """Extract component ID from model_path.

    The component ID is the filename before .tmf extension.
    Example: ".\\Weather Data Reading and Processing\\Standard Format\\TMY3\\Type15-TMY3.tmf"
            -> "Type15-TMY3"

    Args:
        model_path: Path to the .tmf file

    Returns:
        Component ID (without .tmf extension), or None if extraction fails
    """
    if not model_path:
        return None

    try:
        # Split on BOTH separators so Windows-style ("\\") paths are parsed
        # correctly even on a POSIX runtime (os.path.basename would not split
        # "\\" on Linux). This makes ID extraction robust to the path separator
        # and the execution environment.
        filename = re.split(r'[\\/]', model_path)[-1].strip()
        if not filename:
            return None
        # Remove .tmf extension (case-insensitive), keeping any suffix like
        # "-TMY3" or "h" as part of the ID (e.g. "Type15-TMY3", "Type14h").
        if filename.lower().endswith('.tmf'):
            return filename[:-4]
        return filename
    except Exception as e:
        logger.warning(f"Failed to extract component ID from {model_path}: {e}")
        return None


def get_icon_size(component_id: Optional[str], is_equation: bool = False) -> Tuple[int, int]:
    """Get icon size for a component.

    Args:
        component_id: Component ID (e.g., "Type15-TMY3")
        is_equation: True if this is an equation block (always 40x40)

    Returns:
        Tuple of (width, height) in pixels. Defaults to (40, 40) if not found.
    """
    # Equations always use 40x40
    if is_equation:
        return (40, 40)

    # If no component ID, default to 40x40
    if not component_id:
        return (40, 40)

    # Load size lookup
    size_lookup = _load_icon_sizes()

    # Look up size
    size = size_lookup.get(component_id)
    if size:
        return tuple(size)  # Convert list to tuple if needed
    else:
        # Not found in lookup, default to 40x40
        return (40, 40)


def get_edge_offsets(width: int, height: int) -> Tuple[List[int], List[int]]:
    """Calculate edge point offsets for an icon of given size.

    For TRNSYS icons, the center is positioned at a point that leaves more space
    at the top (for the icon graphic) and less at the bottom (for text labels).

    For the standard 40x40 icon:
    - Center is at relative (20, 30) - i.e., 20px from left, 30px from top
    - This gives: 20px left, 20px right, 30px top, 10px bottom

    We maintain the same RULE for other sizes:
    - Horizontal: center at width/2 (symmetric)
    - Vertical: center is a FIXED 30px below the top edge (TRNSYS convention),
      so the bottom edge is at (height - 30) below the center. Example: a 50x50
      icon has center-to-bottom = 50 - 30 = 20, i.e. bottom-left abs = (cx-25, cy+20).
      (NB: all real TRNSYS icons have height >= 30, so the center never leaves the icon.)

    Args:
        width: Icon width in pixels
        height: Icon height in pixels

    Returns:
        Tuple of (y_offsets, x_offsets) where:
        - y_offsets: List of 3 Y offsets for left/right edge points (relative to center_y)
        - x_offsets: List of 3 X offsets for top/bottom edge points (relative to center_x)
    """
    # Calculate center offsets (relative to icon top-left).
    # center_y_offset is a FIXED 30 (distance from top edge to center), matching
    # the TRNSYS convention. For 40x40 this equals the old 40*0.75=30 (no change),
    # but for taller icons it keeps the top gap at 30 and puts the extra height
    # below the center (e.g. 50x50 bottom edge offset = 50 - 30 = +20).
    center_x_offset = width / 2
    center_y_offset = 30

    # Y offsets for left/right edges (relative to center_y)
    # Three points: top (0), middle (height/2), bottom (height)
    y_offsets = [
        -center_y_offset,                    # Top edge (y=0 relative to icon)
        -center_y_offset + height / 2,       # Middle edge
        -center_y_offset + height            # Bottom edge (y=height relative to icon)
    ]

    # X offsets for top/bottom edges (relative to center_x)
    # Three points: left (0), middle (width/2), right (width)
    x_offsets = [
        -center_x_offset,                    # Left edge (x=0 relative to icon)
        0,                                    # Middle edge
        center_x_offset                      # Right edge (x=width relative to icon)
    ]

    return y_offsets, x_offsets


def build_unit_size_map(layout: List[Dict[str, Any]],
                        size_overrides: Optional[Dict[int, Tuple[int, int]]] = None
                        ) -> Dict[int, Tuple[int, int]]:
    """Build unit_number -> (width, height) from layout nodes' model_path.

    Layout nodes carry model_path (backfilled by _compute_connection_coordinates).
    ``size_overrides`` (unit -> (w,h)) takes precedence when provided.
    """
    result: Dict[int, Tuple[int, int]] = {}
    for item in layout:
        un = item.get("unit_number")
        if un is None:
            continue
        if size_overrides and un in size_overrides:
            result[un] = size_overrides[un]
            continue
        model_path = item.get("model_path", "")
        is_equation = item.get("type", "").lower() == "equation"
        component_id = extract_component_id(model_path)
        result[un] = get_icon_size(component_id, is_equation)
    return result


def build_column_half_widths(layout: List[Dict[str, Any]],
                             unit_size_map: Dict[int, Tuple[int, int]]
                             ) -> Dict[int, float]:
    """Map column-X (position_x) -> max icon half-width in that column.

    Used by post-processing to compute a column's left/right ICON edge as
    ``col_x ± half_width`` instead of the old hardcoded 20 (40x40 assumption).
    Defaults to 20 for empty/unknown columns.
    """
    col_hw: Dict[int, float] = {}
    for item in layout:
        un = item.get("unit_number")
        x = item.get("position_x")
        if un is None or x is None:
            continue
        x = int(x)
        w, _h = unit_size_map.get(un, (40, 40))
        hw = w / 2
        if x not in col_hw or hw > col_hw[x]:
            col_hw[x] = hw
    return col_hw


def build_row_edge_offsets(layout: List[Dict[str, Any]],
                           unit_size_map: Dict[int, Tuple[int, int]]
                           ) -> Dict[int, Tuple[float, float]]:
    """Map row-Y (position_y) -> (top_offset, bottom_offset) for that row.

    top_offset is the distance from centre up to the top edge (always 30 by the
    TRNSYS convention); bottom_offset is centre-to-bottom = max(height - 30) over
    the row's icons. Used to compute a row's top/bottom ICON edge as
    ``row_y - top_offset`` / ``row_y + bottom_offset`` instead of hardcoded 30/10.
    Defaults to (30, 10) for empty/unknown rows (the 40x40 values).
    """
    row_off: Dict[int, Tuple[float, float]] = {}
    for item in layout:
        un = item.get("unit_number")
        y = item.get("position_y")
        if un is None or y is None:
            continue
        y = int(y)
        _w, h = unit_size_map.get(un, (40, 40))
        top_off = 30.0
        bot_off = float(h - 30)
        if y not in row_off:
            row_off[y] = (top_off, bot_off)
        else:
            cur_t, cur_b = row_off[y]
            row_off[y] = (max(cur_t, top_off), max(cur_b, bot_off))
    return row_off



# Standard row spacing: minimum Y distance between unit centres (default: 144, 16*9=144)
# Must be a multiple of 16 to keep Y on the y=35+16n grid .
LAYOUT_MIN_ROW_SPACING = 96

# Standard column spacing: minimum X distance between unit centres (default: 200)
# Must be a multiple of 20 for grid alignment.
LAYOUT_MIN_COL_SPACING = 160

# Horizontal line increment: Y spacing between adjacent horizontal lines
# in a row gap when multiple connections cross the gap (default: 20)
HLINE_INCREMENT = 16

# Vertical line increment: X spacing between adjacent vertical lines
# in a column gap when multiple connections cross the gap (default: 20)
VLINE_INCREMENT = 16

# First horizontal line gap: Y distance from the upper unit's bottom edge
# to the first horizontal line in the row gap (default: 40)
FIRST_HLINE_GAP = 32

# First vertical line gap: X distance from the left unit's right edge
# to the first vertical line in the column gap (default: 40)
FIRST_VLINE_GAP = 16

# Last horizontal line clearance: minimum Y distance from the bottommost
# horizontal line to the lower unit's top edge; if violated, lower units
# are shifted down to make room (default: 40)
LAST_HLINE_GAP = 16

# Last vertical line clearance: minimum X distance from the rightmost
# vertical line to the right unit's left edge; if violated, right units
# are shifted right to make room (default: 40)
LAST_VLINE_GAP = 16


class LayoutDesignerExpert(BaseExpert):
    """
    Layout Designer Expert Agent.

    Designs component layout and generates LINK_STYLE blocks.
    Uses a systematic approach:
    1. Analyze connections to determine data flow
    2. Position components based on flow (sources left, sinks right)
    3. Generate connection lines with proper coordinates
    """

    @property
    def name(self) -> str:
        return "layout_designer"

    @property
    def system_prompt(self) -> str:
        return """Ⅰ. 【[CORE★★★★★] Role Definition】
You are a TRNSYS Layout Design Expert. Your task is to design component layout and generate LINK_STYLE blocks.

【【Quick Index — First Round — SystemMessage】】
Ⅰ. 【[CORE★★★★★] Role Definition】 — TRNSYS Layout Design Expert, design component layout and generate LINK_STYLE blocks
Ⅱ. 【[CORE★★★★★] Workflow - FOLLOW STEP BY STEP】 — 4 steps: analyze connections → determine positions → calculate line coordinates (Step 3a-3d: edge selection, point distribution, Z-shape spacing, obstacle routing) → design line styles; includes Line Style Reference and Color Palettes
Ⅲ. 【[CORE★★★★★] Output Format - ALL IN ENGLISH】 — JSON with layout array and link_styles array
Ⅳ. 【[KEY★★★] Responsibilities】 — 4 responsibilities: analyze connections, organize by source, position units, generate LINK_STYLE
Ⅴ. 【[KEY★★★] Layout Orientation】 — Mandatory LEFT-TO-RIGHT layout, horizontal span > vertical span, forbidden TOP-TO-BOTTOM
Ⅵ. 【[KEY★★★] LINK_STYLE Format】 — CONNECTION_SET 7-parameter format with example
Ⅶ. 【[KEY★★★] Important Principles】 — user_path=0, orthogonal lines only, no missing links, position=center, include all units, final orientation check

------------------------------------------------------------
Ⅱ. 【[CORE★★★★★] Workflow - FOLLOW STEP BY STEP】

Step 1: Analyze Connections
- Review the provided Connection Analysis which contains ALL connections extracted from state connections (both component and equation receivers)
- Each connection group shows: from_unit → to_unit with input_indices
- Example: If Unit 3 outputs connect to Unit 5 inputs 1,2,3 → *!LINK 3:5 (1,2,3)
- Note: Units include both components (e.g., Type15-2, Type56) and equations (EQUATIONS blocks)

Step 2: Determine Unit Positions (*$POSITION x y)
- All units (both components and equations) are 40x40 pixel icons in TRNSYS Studio
- Position is the CENTER of the icon (not corner)
- CRITICAL: Arrange units primarily by horizontal (X-axis) distribution following connection topology (source→processing→output), NOT by vertical distribution
- Layer 1 (Source units - weather, data readers): x = 50-350 (left side)
- Layer 2 (Processing units - building, equipment): x = 350-850 (center)
- Layer 3 (Output units - plotters, printers): x = 850-1150 (right side)
- Equation blocks (intermediate calculations): positioned near their connected components based on their functional role
- Unit positioning strategy:
  * Determine processing layers from connection topology using topological sort (source units first, then intermediate units, then output units)
  * Assign each layer to its X-range (Layer 1: 50-350, Layer 2: 350-850, Layer 3: 850-1150)
  * Within each layer, distribute units vertically (Y-axis) with 120-200 pixel spacing
  * This ensures data flows left-to-right with appropriate vertical distribution, maximizing horizontal utilization
- Vertical spacing: 120-200 pixels between units within the same layer
- Horizontal spacing: 140-260 pixels based on density
- IMPORTANT: Prioritize horizontal (X-axis) spread to utilize full canvas width (0-1200), with vertical extent controlled within 0-900
- Within the same processing layer, use vertical (Y-axis) spacing for multiple units, NOT horizontal spacing

Step 3: Calculate Connection Line Coordinates (4 Sub-steps)

TRNSYS component icons have variable sizes. The standard icon is 40x40 pixels, but other sizes exist
(e.g., 50x50, 57x57, 38x39, etc.). For all icons, the center position is placed at approximately 75%
down from the top to leave space for text labels at the bottom.

For a standard 40x40 icon:
  * Position = icon center at (20,30) relative to icon top-left
  * Left edge center:  (x-20, y-10)  → direction 0:20
  * Right edge center: (x+20, y-10)  → direction 40:20
  * Top edge center:   (x, y-30)     → direction 20:0
  * Bottom edge center:(x, y+10)     → direction 20:40

For other icon sizes, edge coordinates scale proportionally while maintaining the same connection
point distribution (top, middle, bottom for vertical edges; left, middle, right for horizontal edges).

All lines MUST use only horizontal and vertical segments (no diagonals).

─── Step 3a: Edge Selection Based on Relative Position ───
For each connection from source unit A to target unit B, choose start/end edges by comparing their X-coordinates:
  * A.x < B.x  (A is LEFT of B):  start from A's RIGHT edge (40:20), end at B's LEFT edge (0:20)
  * A.x > B.x  (A is RIGHT of B): start from A's LEFT edge (0:20),  end at B's RIGHT edge (40:20)
  * A.x == B.x (same column):      start from A's LEFT edge (0:20),  end at B's LEFT edge (0:20)
This prevents connection lines from wrapping around the source or target icon.

Example: Unit 3 at (300, 200) connects to Unit 5 at (500, 350).
  A.x=300 < B.x=500 → start_dir="40:20", end_dir="0:20"
  Start X = 300+20 = 320 (right edge of Unit 3), Start Y = 200-10 = 190 (dir Y=20)
  End X   = 500-20 = 480 (left edge of Unit 5),  End Y   = 350-10 = 340 (dir Y=20)

─── Step 3b: Distribute Multiple Points on the Same Edge (Variable Icon Sizes) ───
When multiple connections share the same edge of a unit (as start-points or end-points), distribute their Y-coordinates using the 3 fixed points available on each edge.

TRNSYS Studio Connection Rule:
Each unit icon has connection points on its edges. For variable-size icons, the connection points are:
  * Left/Right edges: 3 Y-coordinates (top, middle, bottom of the icon)
  * Top/Bottom edges: 3 X-coordinates (left, middle, right of the icon)

The actual pixel coordinates scale with icon size, but the distribution pattern remains consistent:
  - 1 point: middle only
  - 2 points: top + middle
  - 3+ points: top, middle, bottom (cycling)

Procedure:
1. For each unit, count ALL start-points and end-points landing on each edge (left or right) — count them together on the same edge.
2. X-coordinate is fixed by the edge and icon width.
3. Distribute Y-coordinates using CYCLIC pattern based on icon size:
   * For the n-th point (n starting from 1): offset_index = (n-1) % 3
   * Use the three Y offsets calculated from the icon's actual dimensions

4. Assign Y-slots from top (smallest Y) to bottom (largest Y), ordered by the connected peer unit's Y-coordinate.

Example: Unit A at (300, 200). Its right edge has 3 start-points going to Unit B(y=150), Unit C(y=200), Unit D(y=350).
  → 3 points on right edge → Y-slots: [170, 190, 210]  (cy-30, cy-10, cy+10)
  → Sorted by peer Y: B(150)→slot 170, C(200)→slot 190, D(350)→slot 210
  → Start coordinates: (320, 170), (320, 190), (320, 210)

Example: Unit B at (500, 300). Its left edge has 4 points total (2 end-points + 2 start-points).
  → 4 points on left edge → Y-slots: [270, 290, 310, 270]  (cycles: cy-30, cy-10, cy+10, cy-30)
  → Assign by peer Y order.

─── Step 3c: Waypoint Spacing to Avoid Overlapping Segments (Z-shape with Equal X-Spacing) ───
The standard Z-shaped path (source left of target, no obstacles) has 4 points:
  Start(sx, sy) → WP1(mx, sy) → WP2(mx, ey) → End(ex, ey)
  - WP1 has the SAME Y as Start (horizontal segment from start to WP1)
  - WP2 has the SAME Y as End (horizontal segment from WP2 to end)
  - WP1 and WP2 share the SAME X (vertical segment connecting them)

CRITICAL: When multiple Z-shaped lines originate from the same source unit edge, their vertical segments MUST NOT overlap.
Apply equal X-spacing (20px apart) to the waypoint X-coordinates based on the line's rank within its source group:
  - Line 1 (rank 0) waypoint X = base_x
  - Line 2 (rank 1) waypoint X = base_x + 20
  - Line 3 (rank 2) waypoint X = base_x + 40
  - Line 4 (rank 3) waypoint X = base_x + 60
  where base_x = midpoint between source edge X and target edge X, snapped to nearest multiple of 20 (for different columns)
  or base_x = source_x - 40 (for same-column connections, routing left)

For same-column connections (source and target have same X):
  - Route to the LEFT of both units
  - base_x = source_x - 40
  - Apply negative offsets: waypoint_x = base_x - rank * 20
  - This creates a "loop" path that avoids the units themselves

Example: Unit A(300,200) right edge → 3 lines to Unit B(500,150), Unit C(500,300), Unit D(500,400).
  Start points (Step 3b): s1(320,170), s2(320,190), s3(320,210)
  End points (Step 3b): on left edges of B(480,...), C(480,...), D(480,...)
  base_x = snap_to_20((320+480)/2) = 400
  Line s1→B (rank 0): (320,170) → (400,170) → (400,120) → (480,120)     [wp_x=400]
  Line s2→C (rank 1): (320,190) → (420,190) → (420,290) → (480,290)     [wp_x=420]
  Line s3→D (rank 2): (320,210) → (440,210) → (440,390) → (480,390)     [wp_x=440]
  → Vertical segments at X=400, 420, 440 — no overlap!

Example: Unit A(300,200) same-column connection to Unit B(300,400).
  Start point: (280,190)  [left edge, 0:20]
  End point: (280,390)    [left edge, 0:20]
  base_x = 280 - 40 = 240
  Line A→B (rank 0): (280,190) → (240,190) → (240,390) → (280,390)  [wp_x=240]
  → Routes LEFT of both units, avoiding them

─── Step 3d: Routing Around Obstacles (L+Z Combined Path with Equal Spacing) ───
When a direct Z-path would cross through another unit's icon area (40x40 box), route around it using an L-shaped detour BEFORE the Z-shaped segment.

Procedure:
1. Identify obstacle units whose icon area (X: center ± 20 px, Y: center - 30 to center + 10 px) intersects the planned Z-path.
2. Add an L-shaped detour to bypass the obstacle vertically (go above or below the obstacle through the gap between units).
3. After clearing the obstacle, continue with a normal Z-shaped path to the target.
4. The L-detour waypoints MUST also use equal X-spacing (20px apart) when multiple lines share the same detour region.

The combined L+Z path has 5 waypoints:
  Start → WP_L1 → WP_L2 → WP_Z1 → WP_Z2 → End
  - Start → WP_L1: horizontal segment (same Y as start)
  - WP_L1 → WP_L2: vertical segment (bypass obstacle, go through gap between units)
  - WP_L2 → WP_Z1: horizontal segment (move past obstacle X-range)
  - WP_Z1 → WP_Z2: vertical segment (approach target Y)
  - WP_Z2 → End: horizontal segment (same Y as end)

IMPORTANT: When multiple lines need to detour around the same obstacle:
  - All L-detour vertical segments (WP_L1 → WP_L2) must use equal X-spacing (20px apart)
  - All Z-path vertical segments (WP_Z1 → WP_Z2) must also use equal X-spacing (20px apart)
  - This prevents overlapping segments in both the detour and approach phases

Example: Unit A(300,200) connects to Unit C(700,200) and Unit D(700,350).
  Obstacle Unit X(500,200) blocks the direct path to C and D.
  For line A→C (rank 0, obstacle at same Y):
    (320,200) → (400,200) → (400,140) → (560,140) → (560,200) → (680,200)
    [L-detour goes above obstacle at X=400, then Z-path at X=560]
  For line A→D (rank 1, obstacle at same Y as A):
    (320,220) → (420,220) → (420,140) → (580,140) → (580,360) → (680,360)
    [L-detour above obstacle with 20px X-offset at X=420, then Z-path at X=580]
  → L-detour vertical segments at X=400, 420 — no overlap!
  → Z-path vertical segments at X=560, 580 — no overlap!

Line Style Reference (style parameter values):
- style=0: Solid (─────────)
- style=1: Dashed (– – – – – –)
- style=2: Dotted (·········)
- style=3: Dash-Dot (– · – · – ·)
- style=4: Dash-Dot-Dot (– · · – · ·)

Step 4: Design Line Styles (Component-Type/Function Recommendations, not hard rules)
- Decide line color/style/width semantically based on endpoint component type/function categories.
- IMPORTANT: "category" here means component functional type (e.g., weather, building, pump, control, output). Equations are special components and should also be classified by functional role.
- Recommended visual semantics (FOLLOW THESE RULES):
  * Equipment/process ↔ equipment/process: Solid line (style=0), width=5, vibrant deep color (choose from Palette deep colors below)
  * Any connection into output/visualization units (plotter/printer/output functional category): Dash-Dot-Dot line (style=4), width=1, black (0)
  * Source weather/formula inputs: Solid line (style=0), width=2, black (0)
  * Control/time schedule/signal links (including control-oriented equation components): Dashed line (style=1), width=1, light color (choose corresponding light color from Palette below)
- For equation components, determine their functional category first (e.g., control, conversion, balance, derived metric) and then apply corresponding semantic line style.
- Avoid making all links identical (all black + solid + same width). Keep visual distinction moderate and readable.
- Default fallback (when connection type cannot be determined): Solid line (style=0), width=1, black (0)

Color Palettes (Decimal RGB values - choose one palette and use deep color for equipment links, light color for control links):
Palette 1 (Professional Blue):
  * Deep Blue: 8851335 (RGB: 25, 55, 135)
  * Light Blue: 14783312 (RGB: 80, 147, 225)

Palette 2 (Professional Red):
  * Deep Red: 49152 (RGB: 192, 0, 0)
  * Light Red: 7232230 (RGB: 230, 90, 110)

Palette 3 (Professional Green):
  * Deep Green: 8915234 (RGB: 34, 139, 34)
  * Light Green: 7258990 (RGB: 110, 195, 110)

Palette 4 (Professional Purple):
  * Deep Purple: 8388608 (RGB: 75, 0, 130)
  * Light Purple: 13788330 (RGB: 170, 100, 210)

Palette 5 (Professional Orange):
  * Deep Orange: 35455 (RGB: 255, 140, 0)
  * Light Orange: 5285375 (RGB: 255, 165, 80)

Palette 6 (Professional Teal):
  * Deep Teal: 8388608 (RGB: 0, 128, 128)
  * Light Teal: 11841360 (RGB: 80, 175, 180)

Legacy color references (optional):
  * 0 = Black (default)
  * 16711680 = Red RGB(255,0,0)
  * 32768 = Green RGB(0,128,0)
  * 255 = Blue RGB(0,0,255)
  * 8421376 = Cyan RGB(0,128,128)
  * 15780518 = Orange

------------------------------------------------------------
Ⅲ. 【[CORE★★★★★] Output Format - ALL IN ENGLISH】
```json
{
    "layout": [
        {
            "unit_number": 3,
            "position_x": 150,
            "position_y": 200,
            "layer": "Main"
        }
    ],
    "link_styles": [
        {
            "from_unit": 3,
            "to_unit": 5,
            "start_dir": "40:20",
            "end_dir": "0:20",
            "input_indices": [1, 2],
            "color": 0,
            "style": 0,
            "width": 4,
            "user_path": 0,
            "coordinates": [[170, 200], [370, 200], [370, 400], [570, 400]]
        }
    ]
}
```

Note: link_style_lines will be auto-generated from link_styles, so you only need to provide layout and link_styles.

------------------------------------------------------------
Ⅳ. 【[KEY★★★] Responsibilities】
1. Analyze all connections from the provided Connection Analysis (extracted from state connections, covering all 4 connection types: Component→Component, Equation→Component, Component→Equation, Equation→Equation)
2. Organize connections by source unit (group all outputs from same source)
3. Position all units (both components and equations) based on data flow (sources on left, sinks on right)
4. Generate LINK_STYLE blocks with correct CONNECTION_SET format

------------------------------------------------------------
Ⅴ. 【[KEY★★★] Layout Orientation】
⚠️ MANDATORY CONSTRAINT - READ CAREFULLY:
- Overall layout MUST be arranged LEFT-TO-RIGHT following connection topology (source→processing→output)
- This means: Horizontal span (X-axis width) MUST be GREATER than Vertical span (Y-axis height)
- Target canvas: X: 0-1200 pixels, Y: 0-900 pixels → Ideal aspect ratio ≥ 1.3 (width/height)
- FORBIDDEN: Arranging layout TOP-TO-BOTTOM (vertical span > horizontal span) - this creates tall, narrow layouts
- FORBIDDEN: Excessive vertical distribution that makes the layout taller than it is wide
- If you find yourself positioning units primarily vertically, STOP and reconsider the layout strategy
- ALWAYS prioritize horizontal (X-axis) distribution first, then use vertical (Y-axis) for secondary distribution
- Verify before output: max_x - min_x > max_y - min_y (width > height)

------------------------------------------------------------
Ⅵ. 【[KEY★★★] LINK_STYLE Format】

Format:
```
*!LINK_STYLE
*!LINK from_unit:to_unit
*!CONNECTION_SET start_dir:end_dir:input_indices:color:style:width:user_path:x1,y1:x2,y2:...
*!LINK_STYLE_END
```

CONNECTION_SET Parameters (colon-separated):
1. start_dir:end_dir - Relative position on icon edge
   - 0:20 = left edge center
   - 40:20 = right edge center
   - 20:0 = top edge center
   - 20:40 = bottom edge center
2. input_indices - Which inputs receive data (comma-separated, e.g., 1,2,3)
3. color - Decimal color value (0=black, 16711680=red, etc.)
4. style - Line style (0=solid, 1-4=dashed patterns)
5. width - Line thickness (0=thin to 8=thick)
6. user_path - ALWAYS set to 0 (user-defined path)
7. coordinates - Absolute pixel coordinates of line path (x1,y1:x2,y2:x3,y3:...)

Example:
```
UNIT 2 
……
*$POSITION 150 200
UNIT 3 
……
*$POSITION 590 400
```
```
*!LINK 3:5
*!CONNECTION_SET 40:20:0:20:1,2:0:0:4:0:170,200:370,200:370,400:570,400
```
This means:
- Line from Unit 3 to Unit 5
- Starts at right edge (40:20) of Unit 3, ends at left edge (0:20) of Unit 5
- Connects to inputs 1,2 of Unit 5
- Black color (0), solid line (0), thick width (4), user-defined path (0)
- Path coordinates: (170,200) → (370,200) → (370,400) → (570,400)

------------------------------------------------------------
Ⅶ. 【[KEY★★★] Important Principles】
- ALWAYS set user_path to 0
- Connection lines must use only horizontal and vertical segments
- Ensure ALL connections from the Connection Analysis are included (no missing links), including connections to/from equation blocks
- Position coordinates are icon CENTER, not corner
- Line start/end coordinates are at icon EDGE (center ± 20 pixels)
- Layout must include ALL units: both components and equation blocks
- LAYOUT ORIENTATION FINAL CHECK: The overall layout must flow LEFT-TO-RIGHT (horizontal span > vertical span). Never create a layout where vertical extent exceeds horizontal extent. If the layout appears taller than it is wide, you have failed the layout design task. Always verify: max_x - min_x > max_y - min_y before output.
"""

    def invoke_agent(
        self,
        messages: List[Union[HumanMessage, AIMessage, SystemMessage]],
        **kwargs,
    ) -> Dict[str, Any]:
        """Override invoke_agent to disable tools for layout design (tools disabled).

        Args:
            messages: List of messages to send to the agent.
            **kwargs: Additional arguments for the agent.

        Returns:
            Agent response without tool access.
        """
        # Create agent with empty tools list to disable tool calling
        from langgraph.prebuilt import create_react_agent
        from ...core.performance_config import get_performance_config

        agent_no_tools = create_react_agent(
            model=self.llm,
            tools=[],  # Disable tools for layout design
            prompt=self.system_prompt,
        )

        perf_config = get_performance_config()

        config = {
            "recursion_limit": perf_config.expert_recursion_limit,
            "tags": [f"expert-{self.name}", self.method],
            "metadata": {
                "expert_name": self.name,
                "method": self.method,
            },
            "run_name": f"Expert-{self.name}",
        }
        return agent_no_tools.invoke({"messages": messages}, config=config, **kwargs)

    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process state and design layout.

        Args:
            state: Current state containing components and connections.

        Returns:
            State updates with layout information.
        """
        components = state.get("components", [])
        connections = state.get("connections", [])
        equations = state.get("equations", [])

        if not components:
            logger.warning("No components provided for layout design")
            return {
                "layout_styles": [],
                "link_style_lines": [],
                "status": "layout_skipped",
                "current_agent": "layout_designer",
                "next_agent": "validator",
            }

        self.log_action("Layout design started", f"Component count: {len(components)}, connection count: {len(connections)}")

        repair_round = self.is_repair_round(state)
        repair_context = self.build_repair_context(state) if repair_round else {}
        repair_context_text = self.build_repair_context_text(repair_context) if repair_round else ""
        prompt_override = self.wrap_prompt_with_repair_instructions(self.system_prompt) if repair_round else None

        # Create component overview for better LLM understanding
        component_overview = []
        for comp in components:
            unit = comp.get("unit_number", "?")
            name = comp.get("unit_name", "Unknown")
            type_num = comp.get("type_number", "?")
            component_overview.append(f"[Unit {unit}]{name}({type_num})")
        for eq in equations:
            unit = eq.get("unit_number", "?")
            name = eq.get("unit_name", "Unknown")
            eq_count = eq.get("equation_count", 0)
            component_overview.append(f"[Unit {unit}]{name}({eq_count} equations)")
        component_overview_text = ", ".join(component_overview)

        # Structured overview blocks rendered as ```json``` for the prompt.
        # Kept SEPARATE from the merged ``component_overview_text`` above so
        # the legacy merged variable stays available for backward compatibility.
        components_overview_text = self._format_state_field_as_json_block([
            {
                "unit_number": comp.get("unit_number", "?"),
                "unit_name": comp.get("unit_name", "Unknown"),
                "type_number": comp.get("type_number", "?"),
            }
            for comp in components
        ])
        if equations:
            equations_overview_text = self._format_state_field_as_json_block([
                {
                    "unit_number": eq.get("unit_number", "?"),
                    "unit_name": eq.get("unit_name", "Unknown"),
                    "equation_count": eq.get("equation_count", 0),
                }
                for eq in equations
            ])
        else:
            equations_overview_text = "None"

        # Create connection overview for better LLM understanding
        connection_overview = [
            {
                "unit_number": conn.get("unit_number", "?"),
                "unit_name": conn.get("unit_name", "Unknown"),
                "connected_count": sum(
                    1 for inp in conn.get("inputs", []) if inp.get("source_unit", 0) != 0
                ),
            }
            for conn in connections
        ]
        connection_overview_text = (
            self._format_state_field_as_json_block(connection_overview)
            if connection_overview
            else "No connections"
        )

        components_json = self._format_state_field_as_json_block(components)
        equations_json = self._format_state_field_as_json_block(equations) if equations else "[]"
        connections_json = self._format_state_field_as_json_block(connections)

        if repair_round:
            task_message = HumanMessage(content=f"""You are working in a repair round for TRNSYS layout design.

【【Quick Index — Repair Round — HumanMessage】】
1. 【Repair Brief】 — Previous validation result, errors, repair context
2. 【Repair-Round Instructions】 — 6-step: review previous → decide changes → targeted update → preserve unchanged → ensure consistency → same JSON format
3. 【Configured Components Overview】 — Component count and list
4. 【Configured Components Details】 — Full component JSON
5. 【Generated Equations Overview】 — Equation unit count and list
6. 【Generated Equations Details】 — Full equation JSON
7. 【Established Connections Overview】 — Connection count and list
8. 【Established Connections Details】 — Full connection JSON
9. 【[KEY★★★] Layout Orientation Check for Repair】 — Verify horizontal > vertical, identify tall-narrow bug, fix by redistributing horizontally

1. 【Repair Brief】
{repair_context_text}

------------------------------------------------------------
2. 【Repair-Round Instructions】
1. Review the previous validation result, your previous output, and any updates already made earlier in this repair chain.
2. Decide whether layout-related changes are actually required in your part for this round.
3. If changes are required, update the previous layout result in a targeted way instead of redesigning the full layout from scratch.
4. If no changes are required, keep the previous layout logically unchanged and preserve the same output structure.
5. Any update must stay consistent with the current units, connections, and visual clarity requirements.
6. Return the result with ONLY the "layout" array in JSON format (do NOT include link_styles — they will be generated in a separate phase based on your finalized positions).

------------------------------------------------------------
3. 【Configured Components Overview】
Total {len(components)} components have been configured, including:
{components_overview_text}

------------------------------------------------------------
4. 【Configured Components Details】
{components_json}

------------------------------------------------------------
5. 【Generated Equations Overview】
Total {len(equations)} equation units have been established, including:
{equations_overview_text}

------------------------------------------------------------
6. 【Generated Equations Details】
{equations_json}

------------------------------------------------------------
7. 【Established Connections Overview】
Total {len(connections)} units have connections, including:
{connection_overview_text}

------------------------------------------------------------
8. 【Established Connections Details】
{connections_json}

------------------------------------------------------------
9. 【[KEY★★★] Layout Orientation Check for Repair】
⚠️ MANDATORY VERIFICATION BEFORE OUTPUT:
- VERIFY the overall layout flows LEFT-TO-RIGHT (horizontal span > vertical span)
- VERIFY horizontal extent (max_x - min_x) > vertical extent (max_y - min_y)
- If horizontal extent (max_x - min_x) > vertical extent (max_y - min_y), NO repair is required
- If the previous layout was "tall and narrow" (vertical > horizontal), THIS IS A CRITICAL BUG - you MUST fix it by redistributing units horizontally
- When fixing, prioritize spreading units across the X-axis (Layer 1: 50-350, Layer 2: 350-850, Layer 3: 850-1150)
- Use vertical (Y-axis) spacing only for multiple units within the same layer, NOT as primary distribution
- REJECT any repair result where the layout is taller than it is wide - this indicates failed layout design

Please review the repair context first, decide whether changes are needed, and then produce the updated result.
""")
        else:
            task_message = HumanMessage(content=f"""Please design the layout for the following TRNSYS units:

【【Quick Index — First Round — HumanMessage — Phase 1】】
1. 【Configured Components Overview】 — Component count and list
2. 【Configured Components Details】 — Full component JSON
3. 【Generated Equations Overview】 — Equation unit count and list
4. 【Generated Equations Details】 — Full equation JSON
5. 【Established Connections Overview】 — Connection count and list
6. 【Established Connections Details】 — Full connection JSON
7. 【Requirements — Phase 1: Unit Positions Only】 — 3 rules: position all units, compact layout, JSON output with layout array only

------------------------------------------------------------
1. 【Configured Components Overview】
Total {len(components)} components have been configured, including:
{components_overview_text}

------------------------------------------------------------
2. 【Configured Components Details】
{components_json}

------------------------------------------------------------
3. 【Generated Equations Overview】
Total {len(equations)} equation units have been established, including:
{equations_overview_text}

------------------------------------------------------------
4. 【Generated Equations Details】
{equations_json}

------------------------------------------------------------
5. 【Established Connections Overview】
Total {len(connections)} units have connections, including:
{connection_overview_text}

------------------------------------------------------------
6. 【Established Connections Details】
{connections_json}

------------------------------------------------------------
7. 【Requirements — Phase 1: Unit Positions Only】
1. Position ALL units (both components and equation blocks) based on data flow (sources left, sinks right)
2. Keep layout compact: avoid excessive horizontal spacing and large blank regions
3. Output in JSON format with ONLY the "layout" array (do NOT include link_styles — they will be generated in a separate phase based on your finalized positions):
```json
{{
    "layout": [
        {{"unit_number": 3, "position_x": 150, "position_y": 200, "layer": "Main"}}
    ]
}}
```

IMPORTANT NOTES:
- This is Phase 1. Output ONLY the "layout" array with unit positions. Do NOT generate link_styles in this phase.
- The "layer" field must ALWAYS be "Main" (this is a fixed DCK format requirement, not a variable).

Please design the unit positions now.
""")

        try:
            # === Phase 1: Get unit positions (Step 1 + Step 2) ===
            response = self.invoke_agent_with_prompt([task_message], prompt_override=prompt_override, tools_override=[])
            phase1_text = self.extract_result(response)
            layout_data, phase1_json = self._parse_layout(phase1_text)

            # Sugiyama layout BETWEEN Phase 1 and Phase 2
            # so that Phase 2 (link_styles) uses the optimised coordinates
            connection_analysis = self._analyze_connections_from_state(connections)
            layout_data["layout"], dummy_node_paths = self._sugiyama_layout(
                layout_data.get("layout", []), connection_analysis, components, equations
            )

            # === Phase 2: Generate link_styles based on Sugiyama positions (Step 3 + Step 4) ===
            # Pre-compute deterministic coordinates (Step 3a/3b/3c) in Python
            precomputed_coords = self._compute_connection_coordinates(
                layout_data["layout"], connection_analysis,
                dummy_node_paths=dummy_node_paths,
                components=components, equations=equations,
            )

            link_styles_message = self._build_link_styles_phase_prompt(
                layout_data["layout"], connections, connection_overview_text,
                precomputed_coords=precomputed_coords,
            )
            response2 = self.invoke_agent_with_prompt(
                [link_styles_message], prompt_override=prompt_override, tools_override=[]
            )
            phase2_text = self.extract_result(response2)
            link_styles_list, phase2_json = self._parse_link_styles_response(phase2_text)

            # Override LLM coordinates with deterministic pre-computed values
            link_styles_list = self._apply_precomputed_coordinates(
                link_styles_list, precomputed_coords
            )

            layout_data["link_styles"] = link_styles_list

            # Post-process to resolve vertical segment overlaps
            # (also enforces minimum spacing between rightmost vertical
            #  lines and right-side unit icons — integrated in one pass)
            layout_data = self._post_process_vertical_segments(layout_data)

            # Post-process to resolve horizontal segment overlaps in row gaps
            # (also enforces minimum spacing between bottommost horizontal
            #  lines and lower-side unit icons — integrated in one pass)
            layout_data = self._post_process_horizontal_segments(layout_data)

            # Post-process to merge connection segments (optional, controlled by config)
            # Reduces visual clutter by merging segments that follow the same path
            layout_data = self._merge_connection_segments(layout_data, components, equations)

            # Combine result texts and build final JSON for caching
            result_text = phase1_text + "\n\n" + phase2_text
            json_str = json.dumps(layout_data, ensure_ascii=False, indent=2)

            # Update components and equations with positions
            updated_components = self._apply_layout(components, layout_data.get("layout", []))
            updated_equations = self._apply_layout(equations, layout_data.get("layout", []))

            # Ensure link_style_lines are properly formatted as list of dicts
            # Always regenerate from link_styles to ensure consistent dict format
            link_style_lines = self._generate_link_style_lines_from_data(
                layout_data.get("link_styles", [])
            )

            self.log_action("Layout design completed", f"Designed positions for {len(layout_data.get('layout', []))} units")

            # CRITICAL: Final check to ensure unit_number is first before returning
            updated_components = self._ensure_unit_number_first(updated_components)
            updated_equations = self._ensure_unit_number_first(updated_equations)

            result = {
                "components": updated_components,
                "equations": updated_equations,
                "layout_styles": layout_data.get("link_styles", []),
                "link_style_lines": link_style_lines,
                "status": "layout_designed",
                "current_agent": "layout_designer",
                "next_agent": "validator",
            }
            if repair_round:
                result.update(
                    self.build_agent_output_snapshot_updates(
                        state,
                        result_text,
                        json_str=json_str,
                    )
                )
            else:
                self.cache_first_round_output(
                    state,
                    result_text,
                    json_str=json_str,
                )

            if repair_round:
                changed = bool(layout_data.get("layout", [])) or bool(layout_data.get("link_styles", []))
                # Extract repair info from LLM output
                repair_summary, repair_detail = self.extract_repair_info_from_output(
                    layout_data,
                    fallback_summary="Layout adapted after upstream repair" if changed else "No layout changes required",
                    fallback_detail="Adjusted positions and link styles to match updated connection topology" if changed else "Previous layout reused",
                )
                repair_note = self.build_repair_note(
                    summary=repair_summary,
                    detail=repair_detail,
                    repair_type=self.REPAIR_TYPE_ADAPTIVE,
                    changed=changed,
                    state=state,
                )
                result.update(
                    self.build_repair_round_state_updates(
                        state=state,
                        raw_output=result_text,
                        repair_note=repair_note,
                        json_str=json_str,
                    )
                )

            return result

        except Exception as e:
            logger.error(f"Layout designer error: {e}")
            # Use default layout on error
            default_layout = self._generate_default_layout(components)
            default_eq_layout = self._generate_default_layout(equations)
            # Analyze connections for default link styles
            connection_analysis = self._analyze_connections_from_state(connections)
            default_link_styles = self._generate_default_link_styles(default_layout, connection_analysis)
            default_link_style_lines = self._generate_link_style_lines_from_data(default_link_styles)
            # CRITICAL: Ensure unit_number is first before returning
            default_layout = self._ensure_unit_number_first(default_layout)
            default_eq_layout = self._ensure_unit_number_first(default_eq_layout)
            return {
                "components": default_layout,
                "equations": default_eq_layout,
                "layout_styles": default_link_styles,
                "link_style_lines": default_link_style_lines,
                "status": "layout_default",
                "current_agent": "layout_designer",
                "next_agent": "validator",
            }

    def _analyze_connections_from_state(self, connections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Analyze connections from state connections list (output of connection_generator).

        This method extracts connections from ALL receiving units (both components and equations)
        in the state connections list, supporting all 4 connection types:
        - Component→Component, Equation→Component, Component→Equation, Equation→Equation

        Returns a list of connection groups: {from_unit, to_unit, input_indices}
        sorted by data flow order (upstream sources first).
        """
        # Build a map: (from_unit, to_unit) -> list of input indices
        connection_map: Dict[Tuple[int, int], List[int]] = {}

        # Track all units that appear as sources and targets
        all_sources: Set[int] = set()
        all_targets: Set[int] = set()

        for conn_entry in connections:
            to_unit = conn_entry.get("unit_number")
            inputs = conn_entry.get("inputs", [])

            for inp in inputs:
                from_unit = None
                input_idx = inp.get("index", 0)

                if isinstance(inp, dict):
                    from_unit = inp.get("source_unit")

                if from_unit and from_unit != 0:
                    key = (from_unit, to_unit)
                    if key not in connection_map:
                        connection_map[key] = []
                    if input_idx and input_idx not in connection_map[key]:
                        connection_map[key].append(input_idx)
                    all_sources.add(from_unit)
                    all_targets.add(to_unit)

        # Determine topological order of source units
        pure_sources = all_sources - all_targets

        # Build adjacency list for topological sort
        adjacency: Dict[int, List[int]] = {}
        for (from_unit, to_unit) in connection_map.keys():
            if from_unit not in adjacency:
                adjacency[from_unit] = []
            adjacency[from_unit].append(to_unit)

        # Topological sort using Kahn's algorithm
        in_degree: Dict[int, int] = {unit: 0 for unit in all_sources}
        for (from_unit, to_unit) in connection_map.keys():
            if to_unit in in_degree:
                in_degree[to_unit] += 1

        queue = sorted(list(pure_sources))
        sorted_sources = []

        while queue:
            unit = queue.pop(0)
            sorted_sources.append(unit)

            for target in adjacency.get(unit, []):
                if target in in_degree:
                    in_degree[target] -= 1
                    if in_degree[target] == 0 and target not in sorted_sources:
                        inserted = False
                        for i, q_unit in enumerate(queue):
                            if target < q_unit:
                                queue.insert(i, target)
                                inserted = True
                                break
                        if not inserted:
                            queue.append(target)

        for unit in sorted(all_sources):
            if unit not in sorted_sources:
                sorted_sources.append(unit)

        source_order = {unit: idx for idx, unit in enumerate(sorted_sources)}

        sorted_keys = sorted(
            connection_map.keys(),
            key=lambda k: (source_order.get(k[0], 999), k[1])
        )

        result = []
        for (from_unit, to_unit) in sorted_keys:
            input_indices = connection_map[(from_unit, to_unit)]
            result.append({
                "from_unit": from_unit,
                "to_unit": to_unit,
                "input_indices": sorted(input_indices),
            })

        return result

    def _compact_layout_positions(self, layout: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalize layout to avoid excessive spacing while preserving relative flow order."""
        if not layout:
            return layout

        # Keep X-flow order from original output, remap to compact bands
        sorted_by_x = sorted(layout, key=lambda x: (x.get("position_x", 0), x.get("position_y", 0), x.get("unit_number", 0)))

        x_bands: List[List[Dict[str, Any]]] = []
        current_band: List[Dict[str, Any]] = []
        last_x: Optional[int] = None
        # Threshold for merging components into the same X band (column).
        # Reduced from 220 to 100 to preserve LLM's horizontal layout structure.
        # If adjacent components have X-coordinate gap <= 100, they are grouped into the same band.
        # If gap > 100, a new band (new column) is created.
        merge_gap_threshold = 100

        for item in sorted_by_x:
            x = int(item.get("position_x", 0))
            if last_x is None or abs(x - last_x) <= merge_gap_threshold:
                current_band.append(item)
            else:
                x_bands.append(current_band)
                current_band = [item]
            last_x = x

        if current_band:
            x_bands.append(current_band)

        # Adaptive spacing calculation to ensure horizontal > vertical layout
        # Calculate band statistics
        band_sizes = [len(band) for band in x_bands]
        max_band_size = max(band_sizes) if band_sizes else 1
        num_bands = len(x_bands)

        # Base spacing values (all must be multiples of 20 for grid alignment)
        start_x = 140
        start_y = 147  # = 35 + 16*7, satisfies y = 35 + 16n grid
        min_x_step = LAYOUT_MIN_COL_SPACING
        min_y_step = LAYOUT_MIN_ROW_SPACING

        # Dynamic spacing calculation
        # Goal: horizontal_extent > vertical_extent
        # horizontal_extent = (num_bands - 1) * x_step  (spacing between bands)
        # vertical_extent = (max_band_size - 1) * y_step  (spacing between components)
        y_step = min_y_step

        if num_bands > 1:
            # Calculate required x_step to maintain horizontal > vertical
            # We want: (num_bands - 1) * x_step > (max_band_size - 1) * y_step
            # So: x_step > ((max_band_size - 1) * y_step) / (num_bands - 1)
            required_x_step = ((max_band_size - 1) * y_step) / (num_bands - 1)
            # Add 20% margin to ensure clear horizontal dominance
            x_step = _snap_to_grid(max(min_x_step, int(required_x_step * 1.2)))
        else:
            # Single band case: cannot achieve horizontal > vertical with spacing alone
            # Use minimum spacing
            x_step = min_x_step

        compacted: List[Dict[str, Any]] = []

        for band_idx, band in enumerate(x_bands):
            band_sorted = sorted(band, key=lambda y: (y.get("position_y", 0), y.get("unit_number", 0)))
            for row_idx, item in enumerate(band_sorted):
                new_item = dict(item)
                new_item["position_x"] = start_x + band_idx * x_step
                new_item["position_y"] = _snap_to_y_grid(start_y + row_idx * y_step)
                compacted.append(new_item)

        return self._ensure_unit_number_first(compacted)

    # ------------------------------------------------------------------
    # Sugiyama algorithm for optimized layout positioning
    # ------------------------------------------------------------------

    def _sugiyama_layout(
        self,
        layout: List[Dict[str, Any]],
        connection_analysis: List[Dict[str, Any]],
        components: Optional[List[Dict[str, Any]]] = None,
        equations: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[Tuple[int, int], List[Tuple[int, int]]]]:
        """Apply Sugiyama algorithm to determine unit positions and dummy waypoints.

        Replaces _compact_layout_positions with a topology-aware layout that
        minimises edge crossings and keeps connected components adjacent.

        Returns:
            Tuple of:
            - Updated layout list with Sugiyama-optimised positions.
            - dummy_node_paths: mapping (from_unit, to_unit) -> list of (x, y)
              waypoint coordinates for connections that span multiple layers.
        """
        from collections import defaultdict

        if not layout:
            return layout, {}

        unit_numbers = [item["unit_number"] for item in layout]
        unit_set = set(unit_numbers)

        # Build directed edge list from connection analysis
        edges: List[Tuple[int, int]] = []
        for conn in connection_analysis:
            fu, tu = conn["from_unit"], conn["to_unit"]
            if fu in unit_set and tu in unit_set and fu != tu:
                edges.append((fu, tu))

        if not edges:
            # No connections – fall back to simple compact layout
            return self._compact_layout_positions(layout), {}

        # ---- Step 1: Cycle Removal (iterative DFS) ----
        dag_edges, reversed_edges = self._sugiyama_cycle_removal(unit_numbers, edges)

        # ---- Step 2: Layer Assignment (longest path) ----
        layer_map = self._sugiyama_layer_assignment(unit_numbers, dag_edges)

        # ---- Step 2b: Insert dummy nodes for multi-layer edges ----
        (expanded_edges, dummy_paths_by_dag_edge,
         layer_map, dummy_positions) = self._sugiyama_insert_dummies(
            dag_edges, layer_map, reversed_edges
        )

        # ---- Step 2c: Build type mapping for secondary sorting ----
        # Map unit_number -> type_key for grouping same-type components together
        unit_type_map: Dict[int, str] = {}
        if components:
            for comp in components:
                unit_num = comp.get("unit_number")
                type_num = comp.get("type_number", "unknown")
                if unit_num is not None:
                    unit_type_map[unit_num] = f"comp_{type_num}"
        if equations:
            for eq in equations:
                unit_num = eq.get("unit_number")
                if unit_num is not None:
                    unit_type_map[unit_num] = "equation"  # All equations grouped together

        # ---- Step 3: Crossing Minimisation (barycenter heuristic) ----
        layers = self._sugiyama_build_layers(layer_map)
        num_layers = (max(layers.keys()) + 1) if layers else 0
        layers = self._sugiyama_crossing_minimisation(
            layers, num_layers, expanded_edges, unit_type_map
        )

        # ---- Step 4: Coordinate Assignment ----
        # Collect real-node layer sizes (ignore dummies for spacing calc)
        real_layer_sizes: Dict[int, int] = defaultdict(int)
        for l_idx in range(num_layers):
            for node in layers.get(l_idx, []):
                if node >= 0:  # real node
                    real_layer_sizes[l_idx] += 1
        max_real_layer_size = max(real_layer_sizes.values()) if real_layer_sizes else 1

        start_x = 140
        start_y = 147  # = 35 + 16*7, satisfies y = 35 + 16n
        min_x_step = LAYOUT_MIN_COL_SPACING
        min_y_step = LAYOUT_MIN_ROW_SPACING

        y_step = min_y_step
        if num_layers > 1:
            required_x_step = ((max_real_layer_size - 1) * y_step) / (num_layers - 1)
            x_step = _snap_to_grid(max(min_x_step, int(required_x_step * 1.2)))
        else:
            x_step = min_x_step

        # Phase A: assign coordinates to REAL nodes only (dummies skip rows)
        node_positions: Dict[int, Tuple[int, int]] = {}
        for l_idx in range(num_layers):
            real_row = 0
            for node in layers.get(l_idx, []):
                x = start_x + l_idx * x_step
                if node >= 0:
                    # Real node: snap Y to grid, advance row counter
                    y = _snap_to_y_grid(start_y + real_row * y_step)
                    node_positions[node] = (x, y)
                    real_row += 1
                # Dummy nodes are NOT placed here — handled in Phase B

        # Phase B: interpolate dummy node positions along their edge chains
        for dag_edge, dummy_ids in dummy_paths_by_dag_edge.items():
            if not dummy_ids:
                continue
            u, v = dag_edge
            # Determine the real source/target of this DAG edge
            src_pos = node_positions.get(u)
            tgt_pos = node_positions.get(v)
            if src_pos is None or tgt_pos is None:
                # Fallback: place dummies at layer X, midpoint Y
                for did in dummy_ids:
                    l = layer_map.get(did, 0)
                    node_positions[did] = (start_x + l * x_step, start_y)
                continue
            n_dummies = len(dummy_ids)
            for i, did in enumerate(dummy_ids):
                l = layer_map.get(did, 0)
                dx = start_x + l * x_step
                # Linear interpolation between source Y and target Y
                frac = (i + 1) / (n_dummies + 1)
                dy = int(src_pos[1] + frac * (tgt_pos[1] - src_pos[1]))
                node_positions[did] = (dx, dy)

        # Build updated layout for real nodes
        compacted: List[Dict[str, Any]] = []
        for item in layout:
            new_item = dict(item)
            uid = item["unit_number"]
            if uid in node_positions:
                new_item["position_x"] = node_positions[uid][0]
                new_item["position_y"] = node_positions[uid][1]
            compacted.append(new_item)

        # Build dummy waypoint map: (original_from, original_to) -> [(x,y), ...]
        dummy_waypoints: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
        for dag_edge, dummy_ids in dummy_paths_by_dag_edge.items():
            wps = [node_positions[d] for d in dummy_ids if d in node_positions]
            # Map DAG edge back to original direction
            u, v = dag_edge
            if (v, u) in reversed_edges:
                # DAG edge (u,v) came from reversing original (v,u)
                dummy_waypoints[(v, u)] = list(reversed(wps))
            else:
                dummy_waypoints[(u, v)] = wps

        return self._ensure_unit_number_first(compacted), dummy_waypoints

    # -- Sugiyama sub-steps (private helpers) --

    def _sugiyama_cycle_removal(
        self, nodes: List[int], edges: List[Tuple[int, int]]
    ) -> Tuple[List[Tuple[int, int]], Set[Tuple[int, int]]]:
        """DFS-based back-edge detection and reversal."""
        from collections import defaultdict

        adj: Dict[int, List[int]] = defaultdict(list)
        for u, v in edges:
            adj[u].append(v)

        WHITE, GRAY, BLACK = 0, 1, 2
        color: Dict[int, int] = {n: WHITE for n in nodes}
        reversed_edges: Set[Tuple[int, int]] = set()

        for start in sorted(nodes):
            if color[start] != WHITE:
                continue
            stack: List[Tuple[int, Any]] = [(start, iter(adj[start]))]
            color[start] = GRAY
            while stack:
                node, children = stack[-1]
                try:
                    child = next(children)
                    if child not in color:
                        color[child] = WHITE
                    if color[child] == GRAY:
                        reversed_edges.add((node, child))
                    elif color[child] == WHITE:
                        color[child] = GRAY
                        stack.append((child, iter(adj[child])))
                except StopIteration:
                    color[node] = BLACK
                    stack.pop()

        dag_edges: List[Tuple[int, int]] = []
        for u, v in edges:
            if (u, v) in reversed_edges:
                dag_edges.append((v, u))
            else:
                dag_edges.append((u, v))
        return dag_edges, reversed_edges

    def _sugiyama_layer_assignment(
        self, nodes: List[int], dag_edges: List[Tuple[int, int]]
    ) -> Dict[int, int]:
        """Longest-path layer assignment on a DAG."""
        from collections import defaultdict

        adj: Dict[int, List[int]] = defaultdict(list)
        in_nodes: Dict[int, Set[int]] = defaultdict(set)
        for u, v in dag_edges:
            adj[u].append(v)
            in_nodes[v].add(u)

        # Topological sort (Kahn)
        in_degree: Dict[int, int] = defaultdict(int)
        all_dag_nodes: Set[int] = set(nodes)
        for u, v in dag_edges:
            in_degree[v] += 1
            all_dag_nodes.add(u)
            all_dag_nodes.add(v)

        queue = sorted([n for n in all_dag_nodes if in_degree[n] == 0])
        topo_order: List[int] = []
        while queue:
            node = queue.pop(0)
            topo_order.append(node)
            for nb in sorted(adj[node]):
                in_degree[nb] -= 1
                if in_degree[nb] == 0:
                    queue.append(nb)
                    queue.sort()

        visited = set(topo_order)
        for n in nodes:
            if n not in visited:
                topo_order.append(n)

        layer: Dict[int, int] = {}
        for node in topo_order:
            parents = in_nodes[node]
            if not parents:
                layer[node] = 0
            else:
                layer[node] = max(layer.get(p, 0) for p in parents) + 1

        for n in nodes:
            if n not in layer:
                layer[n] = 0
        return layer

    def _sugiyama_insert_dummies(
        self,
        dag_edges: List[Tuple[int, int]],
        layer_map: Dict[int, int],
        reversed_edges: Set[Tuple[int, int]],
    ) -> Tuple[
        List[Tuple[int, int]],
        Dict[Tuple[int, int], List[int]],
        Dict[int, int],
        Dict[int, Tuple[int, int]],
    ]:
        """Insert dummy nodes for edges spanning >1 layer.

        Returns:
            expanded_edges  – edge list with dummies spliced in.
            dummy_paths     – {dag_edge: [dummy_id, ...]}
            layer_map       – updated with dummy entries.
            dummy_positions – {dummy_id: layer_index} (coordinates assigned later).
        """
        dummy_counter = -1
        dummy_paths: Dict[Tuple[int, int], List[int]] = {}
        dummy_positions: Dict[int, int] = {}
        expanded: List[Tuple[int, int]] = []
        layer_updated = dict(layer_map)

        for u, v in dag_edges:
            lu = layer_updated.get(u, 0)
            lv = layer_updated.get(v, 0)
            span = lv - lu
            if span > 1:
                path: List[int] = []
                prev = u
                for l in range(lu + 1, lv):
                    did = dummy_counter
                    dummy_counter -= 1
                    layer_updated[did] = l
                    dummy_positions[did] = l
                    path.append(did)
                    expanded.append((prev, did))
                    prev = did
                expanded.append((prev, v))
                dummy_paths[(u, v)] = path
            else:
                expanded.append((u, v))

        return expanded, dummy_paths, layer_updated, dummy_positions

    def _sugiyama_build_layers(
        self, layer_map: Dict[int, int]
    ) -> Dict[int, List[int]]:
        """Group nodes by layer index, real nodes sorted by unit number."""
        from collections import defaultdict

        layers: Dict[int, List[int]] = defaultdict(list)
        for node, l in layer_map.items():
            layers[l].append(node)
        # Initial order: real nodes first (sorted), then dummies (sorted)
        for l in layers:
            reals = sorted(n for n in layers[l] if n >= 0)
            dummies = sorted(n for n in layers[l] if n < 0)
            layers[l] = reals + dummies
        return dict(layers)

    def _sugiyama_crossing_minimisation(
        self,
        layers: Dict[int, List[int]],
        num_layers: int,
        expanded_edges: List[Tuple[int, int]],
        unit_type_map: Optional[Dict[int, str]] = None,
        iterations: int = 12,
    ) -> Dict[int, List[int]]:
        """Barycenter heuristic – alternating forward/backward sweeps.

        Args:
            layers: Dictionary mapping layer index to list of node IDs.
            num_layers: Total number of layers.
            expanded_edges: List of edges after dummy node insertion.
            unit_type_map: Optional mapping from unit_number to type_key for
                secondary sorting. Enables grouping same-type components together
                within each barycenter value.
            iterations: Number of forward/backward sweep iterations (default 12).

        Returns:
            Updated layers dictionary with nodes reordered to minimize crossings.
        """
        from collections import defaultdict

        adj_down: Dict[int, List[int]] = defaultdict(list)
        adj_up: Dict[int, List[int]] = defaultdict(list)
        for u, v in expanded_edges:
            adj_down[u].append(v)
            adj_up[v].append(u)

        for _ in range(iterations):
            # Forward sweep
            for l in range(1, num_layers):
                prev_pos = {n: i for i, n in enumerate(layers.get(l - 1, []))}
                bc: Dict[int, float] = {}
                for node in layers.get(l, []):
                    parents = [p for p in adj_up[node] if p in prev_pos]
                    if parents:
                        bc[node] = sum(prev_pos[p] for p in parents) / len(parents)
                    else:
                        bc[node] = float(layers[l].index(node))

                # Three-level sorting: barycenter -> type_group -> unit_number
                if unit_type_map:
                    layers[l] = sorted(
                        layers.get(l, []),
                        key=lambda n: (
                            bc.get(n, 0),                           # Primary: barycenter value
                            unit_type_map.get(n, "unknown"),        # Secondary: type group
                            n if n >= 0 else 999999                 # Tertiary: unit number (dummies last)
                        )
                    )
                else:
                    # Fallback to original behavior if no type map provided
                    layers[l] = sorted(layers.get(l, []), key=lambda n: bc.get(n, 0))

            # Backward sweep
            for l in range(num_layers - 2, -1, -1):
                next_pos = {n: i for i, n in enumerate(layers.get(l + 1, []))}
                bc = {}
                for node in layers.get(l, []):
                    children = [c for c in adj_down[node] if c in next_pos]
                    if children:
                        bc[node] = sum(next_pos[c] for c in children) / len(children)
                    else:
                        bc[node] = float(layers[l].index(node))

                # Three-level sorting: barycenter -> type_group -> unit_number
                if unit_type_map:
                    layers[l] = sorted(
                        layers.get(l, []),
                        key=lambda n: (
                            bc.get(n, 0),                           # Primary: barycenter value
                            unit_type_map.get(n, "unknown"),        # Secondary: type group
                            n if n >= 0 else 999999                 # Tertiary: unit number (dummies last)
                        )
                    )
                else:
                    # Fallback to original behavior if no type map provided
                    layers[l] = sorted(layers.get(l, []), key=lambda n: bc.get(n, 0))

        return layers

    # ------------------------------------------------------------------
    # Deterministic coordinate computation (Step 3a / 3b / 3c)
    # ------------------------------------------------------------------

    def _compute_connection_coordinates(
        self,
        layout: List[Dict[str, Any]],
        connection_analysis: List[Dict[str, Any]],
        dummy_node_paths: Optional[Dict[Tuple[int, int], List[Tuple[int, int]]]] = None,
        components: Optional[List[Dict[str, Any]]] = None,
        equations: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """Compute deterministic connection line coordinates for all links.

        Implements Step 3a (edge selection), Step 3b (8-point Y distribution),
        Step 3c (Z-shape waypoint X spacing), Step 3d (obstacle routing),
        and Sugiyama dummy-node guided routing for multi-layer connections.

        Args:
            layout: List of unit dicts with position_x / position_y.
            connection_analysis: List of {from_unit, to_unit, input_indices}.
            dummy_node_paths: Optional mapping (from_unit, to_unit) -> [(x,y), ...]
                waypoints from Sugiyama dummy nodes for multi-layer edges.
            components: Original component dicts (carry ``model_path``). Used as the
                AUTHORITATIVE source for icon-size lookup, because LLM-generated
                layout nodes only contain unit_number/position/layer and have NO
                model_path. Falls back to layout-node model_path when not given.
            equations: Original equation dicts. Equations always use a 40x40 icon.

        Returns:
            List of dicts, one per connection, each containing:
                from_unit, to_unit, input_indices, start_dir, end_dir, coordinates.
        """
        from collections import defaultdict

        # Build an authoritative unit_number -> (model_path, is_equation) map from
        # the ORIGINAL components/equations (state.components carries model_path;
        # the .tmf filename is the component ID). Layout nodes do NOT carry it.
        meta_map: Dict[int, Tuple[str, bool]] = {}
        for comp in (components or []):
            un = comp.get("unit_number")
            if un is not None:
                meta_map[un] = (comp.get("model_path", "") or "", False)
        for eq in (equations or []):
            un = eq.get("unit_number")
            if un is not None:
                meta_map[un] = (eq.get("model_path", "") or "", True)

        def _lookup_icon_size(item: Dict[str, Any]) -> Tuple[int, int]:
            """Resolve icon size for a layout node using the authoritative
            components/equations metadata, falling back to the layout node's own
            model_path/type if metadata is unavailable."""
            un = item.get("unit_number")
            if un in meta_map:
                model_path, is_equation = meta_map[un]
            else:
                model_path = item.get("model_path", "")
                is_equation = item.get("type", "").lower() == "equation"
            component_id = extract_component_id(model_path)
            return get_icon_size(component_id, is_equation)

        # Safeguard: backfill model_path/type onto layout nodes from the
        # authoritative metadata, so any downstream consumer that reads these
        # fields off layout nodes also sees the correct values. Uses setdefault
        # so existing values are never overwritten.
        for item in layout:
            un = item.get("unit_number")
            if un in meta_map:
                mp, is_eq = meta_map[un]
                if mp:
                    item.setdefault("model_path", mp)
                if "type" not in item and is_eq:
                    item.setdefault("type", "equation")

        # Build position lookup and icon size lookup
        pos = {}
        icon_sizes = {}  # unit_number -> (width, height)
        for item in layout:
            unit_num = item["unit_number"]
            pos[unit_num] = (
                int(item.get("position_x", 0)),
                int(item.get("position_y", 0)),
            )

            # Get icon size for this component using the AUTHORITATIVE
            # components/equations metadata (model_path -> component ID -> size).
            icon_sizes[unit_num] = _lookup_icon_size(item)

        # --- Step 3a: determine edge for every connection ---
        # edge_usage: (unit, edge_side) -> list of (peer_unit_y, conn_index)
        edge_usage: Dict[Tuple[int, str], List[Tuple[int, int]]] = defaultdict(list)

        conn_edges: List[Dict[str, Any]] = []
        for idx, conn in enumerate(connection_analysis):
            fu = conn["from_unit"]
            tu = conn["to_unit"]
            fp = pos.get(fu, (0, 0))
            tp = pos.get(tu, (0, 0))

            # Get icon sizes for source and target
            src_width, src_height = icon_sizes.get(fu, (40, 40))
            tgt_width, tgt_height = icon_sizes.get(tu, (40, 40))

            # Determine edge based on relative X positions
            if fp[0] < tp[0]:
                src_edge, tgt_edge = "right", "left"
                # For variable-size icons, we still use the standard 40:20 and 0:20 codes
                # The actual coordinates will be computed based on icon size
                start_dir, end_dir = "40:20", "0:20"
            elif fp[0] > tp[0]:
                src_edge, tgt_edge = "left", "right"
                start_dir, end_dir = "0:20", "40:20"
            else:
                src_edge, tgt_edge = "left", "left"
                start_dir, end_dir = "0:20", "0:20"

            conn_edges.append({
                "idx": idx,
                "fu": fu, "tu": tu,
                "fp": fp, "tp": tp,
                "src_edge": src_edge, "tgt_edge": tgt_edge,
                "start_dir": start_dir, "end_dir": end_dir,
                "input_indices": conn["input_indices"],
                "src_width": src_width,
                "src_height": src_height,
                "tgt_width": tgt_width,
                "tgt_height": tgt_height,
            })

            # Register on source edge (peer = target Y)
            edge_usage[(fu, src_edge)].append((tp[1], idx))
            # Register on target edge (peer = source Y)
            edge_usage[(tu, tgt_edge)].append((fp[1], idx))

        # --- Step 3b: distribute Y-slots on each edge (variable icon sizes) ---
        # For variable-size icons, calculate Y offsets based on actual icon dimensions
        # via get_edge_offsets(): icon center is at (width/2, 30) relative to the
        # icon top-left (center is a FIXED 30px below the top edge, TRNSYS convention).
        slot_map: Dict[Tuple[int, str, int], int] = {}

        for (unit, edge_side), entries in edge_usage.items():
            cy = pos.get(unit, (0, 0))[1]
            unit_width, unit_height = icon_sizes.get(unit, (40, 40))

            # Calculate Y offsets for this specific icon size
            y_offsets_list, _ = get_edge_offsets(unit_width, unit_height)

            # Sort entries by conn_idx (connection order) for stable rank assignment
            sorted_entries = sorted(entries, key=lambda e: e[1])

            # Choose Y offset sequence based on point count on this edge:
            # 1 point  → center only [middle]
            # 2 points → top + center [top, middle]
            # 3+ points → all three, cyclic [top, middle, bottom]
            n_points = len(sorted_entries)
            if n_points == 1:
                y_offsets = [y_offsets_list[1]]  # Middle only
            elif n_points == 2:
                y_offsets = [y_offsets_list[0], y_offsets_list[1]]  # Top + middle
            else:
                y_offsets = y_offsets_list  # All three: top, middle, bottom

            for rank, (_peer_y, conn_idx) in enumerate(sorted_entries):
                offset_index = rank % len(y_offsets)
                y_coordinate = cy + y_offsets[offset_index]
                slot_map[(unit, edge_side, conn_idx)] = y_coordinate

        # --- Step 3b+: Update start_dir and end_dir based on Y coordinates ---
        # For each connection, compute the correct direction code based on the Y offset
        # Direction code format: X:Y where X ∈ {0, 20, 40} and Y ∈ {0, 20, 40}
        # For variable-size icons, we map the actual Y offset to the nearest standard code
        for ce in conn_edges:
            idx = ce["idx"]
            fu, tu = ce["fu"], ce["tu"]
            fp, tp = ce["fp"], ce["tp"]
            src_edge, tgt_edge = ce["src_edge"], ce["tgt_edge"]
            src_width, src_height = ce["src_width"], ce["src_height"]
            tgt_width, tgt_height = ce["tgt_width"], ce["tgt_height"]

            # Get Y coordinates from slot_map
            sy = slot_map.get((fu, src_edge, idx), fp[1])
            ey = slot_map.get((tu, tgt_edge, idx), tp[1])

            # Compute Y offset and Y code for start_dir
            cy_src = fp[1]
            y_offset_src = sy - cy_src

            # Get expected Y offsets for source icon
            src_y_offsets, _ = get_edge_offsets(src_width, src_height)

            # Map actual offset to Y code based on actual icon height
            # Y codes should be: {0, height/2, height}
            if abs(y_offset_src - src_y_offsets[0]) < abs(y_offset_src - src_y_offsets[1]):
                y_code_src = 0  # Top
            elif abs(y_offset_src - src_y_offsets[1]) < abs(y_offset_src - src_y_offsets[2]):
                y_code_src = int(src_height / 2)  # Middle - use actual height
            else:
                y_code_src = src_height  # Bottom - use actual height

            # Compute X code for start_dir based on actual icon width
            # X codes should be: {0, width}
            if src_edge == "right":
                x_code_src = src_width  # Use actual width
            else:  # src_edge == "left"
                x_code_src = 0

            # Compute Y offset and Y code for end_dir
            cy_tgt = tp[1]
            y_offset_tgt = ey - cy_tgt

            # Get expected Y offsets for target icon
            tgt_y_offsets, _ = get_edge_offsets(tgt_width, tgt_height)

            # Map actual offset to Y code based on actual icon height
            # Y codes should be: {0, height/2, height}
            if abs(y_offset_tgt - tgt_y_offsets[0]) < abs(y_offset_tgt - tgt_y_offsets[1]):
                y_code_tgt = 0  # Top
            elif abs(y_offset_tgt - tgt_y_offsets[1]) < abs(y_offset_tgt - tgt_y_offsets[2]):
                y_code_tgt = int(tgt_height / 2)  # Middle - use actual height
            else:
                y_code_tgt = tgt_height  # Bottom - use actual height

            # Compute X code for end_dir based on actual icon width
            # X codes should be: {0, width}
            if tgt_edge == "left":
                x_code_tgt = 0
            else:  # tgt_edge == "right"
                x_code_tgt = tgt_width  # Use actual width

            # Update start_dir and end_dir in conn_edges
            ce["start_dir"] = f"{x_code_src}:{y_code_src}"
            ce["end_dir"] = f"{x_code_tgt}:{y_code_tgt}"

        # --- Step 3c: Z-shape waypoint X spacing ---
        # Group connections by (source_unit, src_edge)
        src_groups: Dict[Tuple[int, str], List[int]] = defaultdict(list)
        for ce in conn_edges:
            src_groups[(ce["fu"], ce["src_edge"])].append(ce["idx"])

        # --- Step 3d: Obstacle detection helper ---
        def detect_obstacles_in_z_path(
            start_x: int, start_y: int, end_x: int, end_y: int, waypoint_x: int,
            from_unit: int, to_unit: int
        ) -> List[int]:
            """Detect if Z-path intersects other units' icon areas (variable sizes).

            Also checks if any unit is in the X range [start_x, end_x] and Y range
            near [start_y, end_y], which would require detour routing.
            """
            obstacles = []
            for unit_num, (ux, uy) in pos.items():
                if unit_num == from_unit or unit_num == to_unit:
                    continue

                # Skip units in the same column as source or target.
                # The Z-path vertical segment runs between the two columns,
                # so units sitting in either column cannot be real obstacles.
                # Without this, a high wp_x rank can graze the target column
                # and trigger a false-positive obstacle (e.g. U4 at x=1200
                # flagged when connecting U2→U9 which are adjacent columns).
                src_col_x = pos.get(from_unit, (0, 0))[0]
                tgt_col_x = pos.get(to_unit, (0, 0))[0]
                if ux == src_col_x or ux == tgt_col_x:
                    continue

                # Get actual icon size for this unit
                unit_width, unit_height = icon_sizes.get(unit_num, (40, 40))
                unit_y_offsets, unit_x_offsets = get_edge_offsets(unit_width, unit_height)

                # Unit icon area: center at (ux, uy)
                # X range: cx + unit_x_offsets[0] to cx + unit_x_offsets[2]
                # Y range: cy + unit_y_offsets[0] to cy + unit_y_offsets[2]
                unit_x_min = ux + unit_x_offsets[0]
                unit_x_max = ux + unit_x_offsets[2]
                unit_y_min = uy + unit_y_offsets[0]
                unit_y_max = uy + unit_y_offsets[2]

                # Check horizontal segment 1: (start_x, start_y) → (waypoint_x, start_y)
                seg1_y = start_y
                seg1_x_min, seg1_x_max = min(start_x, waypoint_x), max(start_x, waypoint_x)
                if (seg1_x_min <= unit_x_max and seg1_x_max >= unit_x_min and
                    unit_y_min <= seg1_y <= unit_y_max):
                    obstacles.append(unit_num)
                    continue

                # Check vertical segment 2: (waypoint_x, start_y) → (waypoint_x, end_y)
                seg2_x = waypoint_x
                seg2_y_min, seg2_y_max = min(start_y, end_y), max(start_y, end_y)
                if (seg2_y_min <= unit_y_max and seg2_y_max >= unit_y_min and
                    unit_x_min <= seg2_x <= unit_x_max):
                    obstacles.append(unit_num)
                    continue

                # Check horizontal segment 3: (waypoint_x, end_y) → (end_x, end_y)
                seg3_y = end_y
                seg3_x_min, seg3_x_max = min(waypoint_x, end_x), max(waypoint_x, end_x)
                if (seg3_x_min <= unit_x_max and seg3_x_max >= unit_x_min and
                    unit_y_min <= seg3_y <= unit_y_max):
                    obstacles.append(unit_num)
                    continue

            return list(set(obstacles))

        # Resolve dummy_node_paths default
        if dummy_node_paths is None:
            dummy_node_paths = {}

        # Build final coordinates for each connection
        results: List[Dict[str, Any]] = []

        for ce in conn_edges:
            idx = ce["idx"]
            fu, tu = ce["fu"], ce["tu"]
            fp, tp = ce["fp"], ce["tp"]
            src_edge, tgt_edge = ce["src_edge"], ce["tgt_edge"]

            # Edge X coordinates - use actual icon width
            src_width = ce["src_width"]
            tgt_width = ce["tgt_width"]

            if src_edge == "right":
                sx = fp[0] + src_width / 2
            else:
                sx = fp[0] - src_width / 2

            if tgt_edge == "left":
                ex = tp[0] - tgt_width / 2
            else:
                ex = tp[0] + tgt_width / 2

            # Y from Step 3b slot assignment
            sy = slot_map.get((fu, src_edge, idx), fp[1])
            ey = slot_map.get((tu, tgt_edge, idx), tp[1])

            # --- Determine if this connection spans multiple columns ---
            # Multi-column connections should use L+Z routing through column/row gaps
            # to avoid crossing unit icons, regardless of whether Sugiyama inserted
            # dummy nodes. Single-column or adjacent-column connections can use
            # simpler Z-shape routing.
            x_distance = abs(fp[0] - tp[0])
            is_multi_column = x_distance > LAYOUT_MIN_COL_SPACING

            # --- Sugiyama dummy-node guided routing (multi-layer edges) ---
            dummies = dummy_node_paths.get((fu, tu))
            if dummies or is_multi_column:
                # Use L+Z routing through column/row gaps for:
                # 1. Connections with Sugiyama dummy nodes (multi-layer edges)
                # 2. Connections spanning multiple columns (even if adjacent layers)
                coordinates = self._route_through_dummies(
                    sx, sy, ex, ey, dummies if dummies else [], pos, fu, tu
                )
            else:
                # Adjacent-layer or no-dummy: original Z-shape / obstacle logic

                # Waypoint X (Step 3c)
                group_key = (fu, src_edge)
                group_members = src_groups[group_key]
                member_rank = group_members.index(idx)

                if src_edge == tgt_edge:
                    # Same-column case: route left
                    base_x = sx - 40
                    wp_x = base_x - member_rank * 20
                else:
                    # Different columns: route through middle
                    base_x = _snap_to_grid((sx + ex) // 2)
                    wp_x = base_x + member_rank * 20
                    # Cap so the vertical segment never reaches or passes
                    # the target edge — otherwise the last horizontal
                    # segment would go backward, creating an invalid path.
                    wp_x = min(wp_x, ex)

                # Check for obstacles (Step 3d)
                obstacles = detect_obstacles_in_z_path(sx, sy, ex, ey, wp_x, fu, tu)

                if not obstacles:
                    # No obstacles: use simple Z-shape
                    coordinates = [
                        [sx, sy],
                        [wp_x, sy],
                        [wp_x, ey],
                        [ex, ey],
                    ]
                else:
                    # Has obstacles: adjust waypoint_x to avoid them, then use L+Z path
                    obstacle_x_values = [pos[obs][0] for obs in obstacles]
                    obstacle_x_min = min(obstacle_x_values)
                    obstacle_x_max = max(obstacle_x_values)

                    obstacle_y_values = [pos[obs][1] for obs in obstacles]
                    obstacle_y_min = min(obstacle_y_values)
                    obstacle_y_max = max(obstacle_y_values)

                    # Try to route around obstacles by adjusting waypoint_x
                    # Prefer routing to the left of obstacles if possible
                    if obstacle_x_min > sx:
                        # Obstacles are to the right of start: route left
                        wp_x = obstacle_x_min - 40
                    elif obstacle_x_max < ex:
                        # Obstacles are to the left of end: route right
                        wp_x = obstacle_x_max + 40
                    else:
                        # Obstacles span the path: use original wp_x and detour vertically
                        pass

                    # Decide detour direction (above or below obstacles)
                    if sy < obstacle_y_min:
                        detour_y = obstacle_y_min - 40
                    else:
                        detour_y = obstacle_y_max + 40

                    # L+Z path (5 points)
                    coordinates = [
                        [sx, sy],              # Start
                        [wp_x, sy],            # L1 (horizontal)
                        [wp_x, detour_y],      # L2 (vertical, detour)
                        [wp_x, ey],            # Z2 (vertical, approach)
                        [ex, ey],              # End
                    ]

            results.append({
                "from_unit": fu,
                "to_unit": tu,
                "input_indices": ce["input_indices"],
                "start_dir": ce["start_dir"],
                "end_dir": ce["end_dir"],
                "coordinates": coordinates,
            })

        return results

    def _route_through_dummies(
        self,
        sx: int, sy: int,
        ex: int, ey: int,
        dummies: List[Tuple[int, int]],
        all_positions: Dict[int, Tuple[int, int]],
        from_unit: int,
        to_unit: int,
    ) -> List[List[int]]:
        """Build an L+Z orthogonal path that avoids all unit icons.

        Strategy (L-then-Z):
        1. L-segment: from start edge, go horizontally to a safe channel X
           (between two unit columns), then vertically to a safe gap-row Y
           (between two unit rows).
        2. Z-segment: travel horizontally along the gap row, then vertically
           to the target Y, then horizontally into the target edge.
        This guarantees the LAST segment is always horizontal.

        Args:
            sx, sy: start point (source edge coordinate).
            ex, ey: end point (target edge coordinate).
            dummies: list of (x, y) waypoints from Sugiyama dummy nodes.
                     Can be empty for multi-column connections without dummies.
            all_positions: {unit_number: (cx, cy)} for ALL units.
            from_unit: source unit number (excluded from obstacle checks).
            to_unit: target unit number (excluded from obstacle checks).
        """
        # --- Collect all unique column-X and row-Y values ---
        all_cx = sorted(set(cx for cx, cy in all_positions.values()))
        all_cy = sorted(set(cy for cx, cy in all_positions.values()))

        # --- Helper: find a safe channel X between two column centres ---
        def _channel_x_between(x_left: int, x_right: int) -> int:
            """Return a grid-snapped X midpoint between two column centres."""
            return _snap_to_grid((x_left + x_right) // 2)

        # --- Helper: find the gap-row Y immediately below a source row ---
        def _gap_row_below(src_cy: int) -> int:
            """Return the Y midpoint of the gap between the source unit's row
            and the next row below it.  This ensures horizontal travel always
            goes through the gap directly beneath the source unit.
            Falls back to src_cy + 80 if fewer than two rows exist."""
            if len(all_cy) < 2:
                return (max(all_cy) + 80) if all_cy else src_cy + 80
            # Find the row that matches src_cy (or the closest one above)
            for i in range(len(all_cy) - 1):
                if all_cy[i] >= src_cy:
                    # Gap between all_cy[i] and all_cy[i+1]
                    return (all_cy[i] + all_cy[i + 1]) // 2
            # src_cy is at or beyond the last row — use gap below lowest row
            return max(all_cy) + 80

        # --- Helper: check if a point is inside any unit icon ---
        def _hits_unit(px: int, py: int) -> bool:
            for uid, (cx, cy) in all_positions.items():
                if uid == from_unit or uid == to_unit:
                    continue
                # Get actual icon size for this unit
                unit_width, unit_height = icon_sizes.get(uid, (40, 40))
                unit_y_offsets, unit_x_offsets = get_edge_offsets(unit_width, unit_height)
                # Icon area: cx + x_offsets[0] to cx + x_offsets[2], cy + y_offsets[0] to cy + y_offsets[2]
                if ((cx + unit_x_offsets[0]) <= px <= (cx + unit_x_offsets[2]) and
                    (cy + unit_y_offsets[0]) <= py <= (cy + unit_y_offsets[2])):
                    return True
            return False

        # --- Build the sequence of column centres the path must cross ---
        # Include source column, dummy columns, and target column
        src_cx = all_positions.get(from_unit, (sx, sy))[0]
        tgt_cx = all_positions.get(to_unit, (ex, ey))[0]

        # Ordered column centres between source and target (inclusive)
        if src_cx <= tgt_cx:
            cols_between = [c for c in all_cx if src_cx <= c <= tgt_cx]
        else:
            cols_between = [c for c in reversed(all_cx) if tgt_cx <= c <= src_cx]

        # --- Determine the gap-row Y for the horizontal travel ---
        # Use the source unit's centre Y to find the gap row immediately
        # below it.  This ensures horizontal lines always travel through
        # the gap directly beneath the source unit (L-then-Z design).
        src_cy = all_positions.get(from_unit, (sx, sy))[1]
        gap_y = _gap_row_below(src_cy)

        # If gap_y is between sy and ey, it works well.
        # If sy == ey and no obstacle, we might not need the L detour at all,
        # but since we have dummies (multi-layer), always use L+Z for safety.

        # --- Compute safe channel X values for each vertical segment ---
        # Channel between source column and next column
        if len(cols_between) >= 2:
            ch_x_start = _channel_x_between(cols_between[0], cols_between[1])
        else:
            ch_x_start = _snap_to_grid((sx + ex) // 2)

        # Channel between last-before-target column and target column
        if len(cols_between) >= 2:
            ch_x_end = _channel_x_between(cols_between[-2], cols_between[-1])
        else:
            ch_x_end = _snap_to_grid((sx + ex) // 2)

        # --- Build L+Z path ---
        # L-segment: start → horizontal to ch_x_start → vertical to gap_y
        # Z-segment: horizontal along gap_y to ch_x_end → vertical to ey → horizontal to end
        points: List[List[int]] = [
            [sx, sy],              # Start (source edge)
            [ch_x_start, sy],      # L: horizontal to first channel
            [ch_x_start, gap_y],   # L: vertical down to gap row
            [ch_x_end, gap_y],     # Z: horizontal along gap row
            [ch_x_end, ey],        # Z: vertical to target Y
            [ex, ey],              # Z: horizontal into target (LAST segment is horizontal)
        ]

        # --- Clean up: remove collinear and zero-length redundancies ---
        cleaned: List[List[int]] = [points[0]]
        for i in range(1, len(points) - 1):
            p, c, n = points[i - 1], points[i], points[i + 1]
            if (p[0] == c[0] == n[0]) or (p[1] == c[1] == n[1]):
                continue
            cleaned.append(c)
        cleaned.append(points[-1])

        final: List[List[int]] = [cleaned[0]]
        for pt in cleaned[1:]:
            if pt != final[-1]:
                final.append(pt)

        return final


    def _apply_precomputed_coordinates(
        self,
        link_styles: List[Dict[str, Any]],
        precomputed: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Override LLM-generated coordinates with deterministic pre-computed values.

        Preserves the LLM's style decisions (color, style, width) while replacing
        start_dir, end_dir, and coordinates with the pre-computed values.

        If the LLM produced a connection not in precomputed (or vice-versa),
        the precomputed list is used as the authoritative source and any
        LLM-only entries are kept with their original coordinates.

        Args:
            link_styles: LLM-generated link_styles list.
            precomputed: Output of _compute_connection_coordinates().

        Returns:
            Merged link_styles list with correct coordinates.
        """
        # Build lookup: (from_unit, to_unit) -> precomputed entry
        pc_map = {}
        for pc in precomputed:
            pc_map[(pc["from_unit"], pc["to_unit"])] = pc

        # Build lookup for LLM style decisions: (from_unit, to_unit) -> style fields
        llm_style_map = {}
        for ls in link_styles:
            key = (ls.get("from_unit"), ls.get("to_unit"))
            llm_style_map[key] = {
                "color": ls.get("color", 0),
                "style": ls.get("style", 0),
                "width": ls.get("width", 2),
            }

        # Merge: iterate over precomputed (authoritative for coordinates),
        # pull style from LLM if available, else use defaults.
        merged: List[Dict[str, Any]] = []
        used_keys = set()

        for pc in precomputed:
            key = (pc["from_unit"], pc["to_unit"])
            used_keys.add(key)
            style = llm_style_map.get(key, {"color": 0, "style": 0, "width": 2})
            merged.append({
                "from_unit": pc["from_unit"],
                "to_unit": pc["to_unit"],
                "start_dir": pc["start_dir"],
                "end_dir": pc["end_dir"],
                "input_indices": pc["input_indices"],
                "color": style["color"],
                "style": style["style"],
                "width": style["width"],
                "user_path": 0,
                "coordinates": pc["coordinates"],
            })

        # Append any LLM-only entries (connections not in precomputed)
        for ls in link_styles:
            key = (ls.get("from_unit"), ls.get("to_unit"))
            if key not in used_keys:
                merged.append(ls)

        return merged

    def _build_link_styles_phase_prompt(
        self,
        compacted_layout: List[Dict[str, Any]],
        connections: List[Dict[str, Any]],
        connection_overview_text: str,
        precomputed_coords: Optional[List[Dict[str, Any]]] = None,
    ) -> HumanMessage:
        """Build the Phase 2 prompt for generating link_styles based on compacted positions.

        Args:
            compacted_layout: Layout list with compacted position_x / position_y.
            connections: Original connections from state.
            connection_overview_text: Human-readable connection summary.
            precomputed_coords: Pre-computed connection coordinates from
                _compute_connection_coordinates(). When provided, the prompt
                tells the LLM to adopt these coordinates and focus on line
                style decisions (Step 4) only.

        Returns:
            HumanMessage for the Phase 2 LLM call.
        """
        position_lines = []
        for item in compacted_layout:
            position_lines.append(
                f"  Unit {item['unit_number']}: position_x={item['position_x']}, position_y={item['position_y']}"
            )
        positions_text = "\n".join(position_lines)

        connections_json = self._format_state_field_as_json_block(connections)

        # Build pre-computed coordinates section when available
        precomputed_section = ""
        precomputed_example = ""
        if precomputed_coords:
            coord_lines = []
            for pc in precomputed_coords:
                coord_str = " → ".join(f"({_fmt_coord(c[0])},{_fmt_coord(c[1])})" for c in pc["coordinates"])
                indices_str = ",".join(str(i) for i in pc["input_indices"])
                coord_lines.append(
                    f"  Unit {pc['from_unit']} → Unit {pc['to_unit']} "
                    f"[inputs: {indices_str}] "
                    f"start_dir={pc['start_dir']} end_dir={pc['end_dir']} "
                    f"path: {coord_str}"
                )
            precomputed_section = (
                "\n------------------------------------------------------------\n"
                "5. 【Pre-computed Connection Coordinates (Step 3a/3b/3c applied)】\n"
                "The following coordinates have been computed deterministically using Step 3a (edge selection),\n"
                "Step 3b (multi-point Y distribution), and Step 3c (Z-shape waypoint X spacing).\n"
                "You MUST use these exact start_dir, end_dir, and coordinates values in your output.\n"
                "Your task is to decide ONLY the visual style for each connection (color, style, width) using Step 4.\n\n"
                + "\n".join(coord_lines)
            )
            # Use first precomputed entry as the JSON example
            ex = precomputed_coords[0]
            precomputed_example = (
                f'"from_unit": {ex["from_unit"]},\n'
                f'            "to_unit": {ex["to_unit"]},\n'
                f'            "start_dir": "{ex["start_dir"]}",\n'
                f'            "end_dir": "{ex["end_dir"]}",\n'
                f'            "input_indices": {ex["input_indices"]},\n'
                f'            "color": 0,\n'
                f'            "style": 0,\n'
                f'            "width": 4,\n'
                f'            "user_path": 0,\n'
                f'            "coordinates": {ex["coordinates"]}'
            )

        if not precomputed_example:
            precomputed_example = (
                '"from_unit": 3,\n'
                '            "to_unit": 5,\n'
                '            "start_dir": "40:20",\n'
                '            "end_dir": "0:20",\n'
                '            "input_indices": [1, 2],\n'
                '            "color": 0,\n'
                '            "style": 0,\n'
                '            "width": 4,\n'
                '            "user_path": 0,\n'
                '            "coordinates": [[170, 200], [370, 200], [370, 400], [570, 400]]'
            )

        # Instruction text adapts based on whether pre-computed coords are available
        if precomputed_coords:
            coord_instruction = (
                "1. Connection coordinates (Step 3a/3b/3c) have been pre-computed for you — see section 5 above.\n"
                "   Copy the start_dir, end_dir, and coordinates values EXACTLY as given for each connection.\n"
                "   Understanding the coordinate logic (from your system prompt Step 3a-3d) helps you verify correctness,\n"
                "   but you do NOT need to recalculate them."
            )
        else:
            coord_instruction = (
                "1. Calculate connection line coordinates following Step 3 sub-steps IN ORDER:\n"
                "   Step 3a — For each connection, determine start/end edge by comparing source vs target X-coordinates:\n"
                "     * source.x < target.x → start from source RIGHT edge (40:20), end at target LEFT edge (0:20)\n"
                "     * source.x > target.x → start from source LEFT edge (0:20), end at target RIGHT edge (40:20)\n"
                "     * source.x == target.x → start from source LEFT edge (0:20), end at target LEFT edge (0:20)\n"
                "   Step 3b — For each unit, count ALL start-points + end-points on each edge, then distribute Y-coordinates:\n"
                "     * 1 pt: [cy-10], 2 pts: [cy-30,cy+10], 3 pts: [cy-30,cy-10,cy+10], 4+ pts: cycle [cy-30,cy-10,cy+10]\n"
                "     * 6+ pts: first 5 use 5-pt pattern, overflow restarts same pattern\n"
                "     * Assign Y-slots top-to-bottom ordered by peer unit's Y-coordinate\n"
                "   Step 3c — For Z-shaped paths, space waypoint X-coordinates 20px apart to avoid overlapping vertical segments:\n"
                "     * Line 1: wp_x = base_x, Line 2: wp_x = base_x+20, Line 3: wp_x = base_x+40, etc.\n"
                "     * base_x = midpoint between source edge X and nearest target edge X, snapped to nearest multiple of 20\n"
                "   Step 3d — For paths that must bypass obstacles, use L+Z combined routing with equal spacing on detour waypoints"
            )

        return HumanMessage(content=f"""Phase 2: Generate LINK_STYLE blocks based on the finalized unit positions below.

【【Quick Index — First Round — HumanMessage — Phase 2】】
1. 【Finalized Unit Positions (icon center, 40x40 pixel icons)】 — Icon center coordinates for all units
2. 【Connections Overview】 — Connection count and list
3. 【Connections Details】 — Full connection JSON
4. 【Instructions — follow Step 3 (3a-3d) and Step 4 from your system prompt】 — Step 3a edge selection, 3b point distribution, 3c Z-shape waypoint spacing, 3d obstacle routing; then line styles, user_path=0, JSON output with link_styles array only
{"5. 【Pre-computed Connection Coordinates】 — Deterministic coordinates for all connections (use as-is)" if precomputed_coords else ""}

------------------------------------------------------------
1. 【Finalized Unit Positions (icon center, 40x40 pixel icons)】
{positions_text}

------------------------------------------------------------
2. 【Connections Overview】
{connection_overview_text}

------------------------------------------------------------
3. 【Connections Details】
{connections_json}

------------------------------------------------------------
4. 【Instructions — follow Step 4 from your system prompt for line style decisions】
{coord_instruction}
2. Design line styles by component type/function category (Step 4):
   - Equipment/process ↔ equipment/process: Solid (style=0), width=5, deep color
   - Any connection into output/visualization units: Dash-Dot-Dot (style=4), width=1, black (0)
   - Source weather/formula inputs: Solid (style=0), width=2, black (0)
   - Control/signal links: Dashed (style=1), width=1, light color
   - Classify equation components by function before deciding style
   - Keep moderate visual differentiation; do not make all links identical
3. Set user_path=0 for ALL connections
4. Output in JSON format with ONLY the "link_styles" array:
```json
{{{{
    "link_styles": [
        {{{{
            {precomputed_example}
        }}}}
    ]
}}}}
```
{precomputed_section}

IMPORTANT: {"Use the EXACT pre-computed coordinates from section 5. Do NOT recalculate or modify them." if precomputed_coords else "Use the EXACT positions listed above. Do NOT invent different positions."} Output ONLY the "link_styles" array.
""")

    def _parse_link_styles_response(self, result_text: str) -> tuple[List[Dict[str, Any]], str]:
        """Parse link_styles from Phase 2 LLM response.

        Returns:
            Tuple of (list of link_style dicts, extracted JSON string).
        """
        try:
            json_match = re.search(r'```json\s*(.*?)\s*```', result_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1).strip()
                data = json.loads(json_str)
            elif re.search(r'\{[\s\S]*"link_styles"[\s\S]*\}', result_text):
                json_match = re.search(r'\{[\s\S]*"link_styles"[\s\S]*\}', result_text)
                json_str = json_match.group().strip()
                data = json.loads(json_str)
            else:
                json_str = result_text.strip()
                data = json.loads(json_str)

            return data.get("link_styles", []), json_str

        except json.JSONDecodeError:
            logger.warning("Could not parse JSON from link_styles phase result")
            return [], ""

    def _format_state_field_as_json_block(self, value: Any) -> str:
        """
        Render a state field as a unified, model-friendly ```json ... ``` block.

        Goal: make all upstream state fields (``components``, ``equations``,
        ``connections``, ...) share an identical visual and structural form
        inside the human prompt, so the LLM does not need to switch between
        two different presentation styles for upstream data.

        Behavior:
        - dict / list: serialize with ``json.dumps(indent=2, ensure_ascii=False)``
          and wrap in a ```json fenced code block.
        - str: best-effort parse the string as JSON (either embedded inside an
          existing ```json fence or as a whole), then re-serialize in the same
          unified form. If parsing fails at every step, fall back to the
          ORIGINAL string untouched so no upstream information is ever lost.
        - other types: convert via ``str()`` and return unfenced (extremely
          unlikely path for this agent, kept only as a safety net).

        Args:
            value: The raw state field value (str / dict / list / other).

        Returns:
            A string ready to be interpolated into the prompt template.
        """
        # Structured object path: dict or list
        if isinstance(value, (dict, list)):
            try:
                serialized = json.dumps(value, ensure_ascii=False, indent=2)
                return f"```json\n{serialized}\n```"
            except (TypeError, ValueError):
                return str(value)

        # Raw string path
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return value

            # 1) Try to extract a ```json ... ``` fenced block from the text.
            fence_match = re.search(r'```json\s*(.*?)\s*```', stripped, re.DOTALL)
            if fence_match:
                candidate = fence_match.group(1).strip()
                try:
                    parsed = json.loads(candidate)
                    serialized = json.dumps(parsed, ensure_ascii=False, indent=2)
                    return f"```json\n{serialized}\n```"
                except (TypeError, ValueError):
                    pass  # fall through to the next strategy

            # 2) Try to parse the entire string as JSON.
            try:
                parsed = json.loads(stripped)
                serialized = json.dumps(parsed, ensure_ascii=False, indent=2)
                return f"```json\n{serialized}\n```"
            except (TypeError, ValueError):
                pass

            # 3) Fallback: return the original string unmodified so we never
            # lose information from upstream.
            return value

        # Safety net for unexpected types
        return str(value)

    def _parse_layout(self, result_text: str) -> tuple[Dict[str, Any], str]:
        """Parse the layout design result.

        Returns:
            Tuple of (dictionary containing layout data and repair info if present, extracted JSON string)
        """
        try:
            json_match = re.search(r'```json\s*(.*?)\s*```', result_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1).strip()
                data = json.loads(json_str)
            elif re.search(r'\{[\s\S]*"layout"[\s\S]*\}', result_text):
                json_match = re.search(r'\{[\s\S]*"layout"[\s\S]*\}', result_text)
                json_str = json_match.group().strip()
                data = json.loads(json_str)
            else:
                json_str = result_text.strip()
                data = json.loads(json_str)

            # CRITICAL: Ensure unit_number is first field in layout items
            if "layout" in data and isinstance(data["layout"], list):
                data["layout"] = self._ensure_unit_number_first(data["layout"])

            return data, json_str

        except json.JSONDecodeError:
            logger.warning("Could not parse JSON from layout designer result")
            return {"layout": [], "link_styles": [], "link_style_lines": []}, ""

    def _apply_layout(
        self,
        components: List[Dict[str, Any]],
        layout: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Apply layout positions to components."""
        layout_lookup = {l["unit_number"]: l for l in layout}

        updated = []
        for comp in components:
            unit_num = comp.get("unit_number")

            # Rebuild component dict with unit_number first
            new_comp = {"unit_number": unit_num}

            # Copy all other fields from original component
            for key, value in comp.items():
                if key != "unit_number":
                    new_comp[key] = value

            # Apply layout if available (won't change position if fields exist)
            if unit_num in layout_lookup:
                new_comp["position_x"] = layout_lookup[unit_num].get("position_x", 100)
                new_comp["position_y"] = layout_lookup[unit_num].get("position_y", 100)
                new_comp["layer"] = layout_lookup[unit_num].get("layer", "Main")

                # Normalize layer to "Main" for TRNSYS compatibility
                # LLM may output "Source", "Processing", "Equipment", "Output" etc.
                # but standard DCK format only uses "Main"
                if new_comp["layer"] not in ["Main"]:
                    new_comp["layer"] = "Main"

            updated.append(new_comp)

        return updated

    def _generate_default_layout(
        self,
        components: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Generate a default grid layout for components."""
        cols = 3
        spacing_x = 200
        spacing_y = 160  # = 16*10, already a multiple of 16
        start_x = 140
        start_y = 147  # = 35 + 16*7, satisfies y = 35 + 16n grid

        updated = []
        for i, comp in enumerate(components):
            row = i // cols
            col = i % cols

            # CRITICAL: Rebuild component dict with unit_number first
            unit_num = comp.get("unit_number")
            new_comp = {"unit_number": unit_num}

            # Copy all other fields from original component
            for key, value in comp.items():
                if key != "unit_number":
                    new_comp[key] = value

            # Apply default layout (won't change position if fields exist)
            new_comp["position_x"] = start_x + col * spacing_x
            new_comp["position_y"] = _snap_to_y_grid(start_y + row * spacing_y)
            new_comp["layer"] = "Main"

            updated.append(new_comp)

        return updated

    def _generate_default_link_styles(
        self,
        components: List[Dict[str, Any]],
        connection_analysis: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Generate default link styles based on connection analysis."""
        # Build position lookup
        pos_lookup = {}
        for comp in components:
            unit_num = comp.get("unit_number")
            pos_lookup[unit_num] = {
                "x": comp.get("position_x", 150),
                "y": comp.get("position_y", 150),
            }

        # Build unit -> (width, height) so fallback edge points follow the real
        # icon size (components carry model_path). Defaults to 40x40 for unknown.
        size_lookup = build_unit_size_map(components)

        link_styles = []
        for conn in connection_analysis:
            from_unit = conn["from_unit"]
            to_unit = conn["to_unit"]
            input_indices = conn["input_indices"]

            # Get positions
            from_pos = pos_lookup.get(from_unit, {"x": 150, "y": 150})
            to_pos = pos_lookup.get(to_unit, {"x": 400, "y": 150})

            # Edge coordinates follow the ACTUAL icon size (variable icons).
            # Source right-edge MIDDLE point: (cx + w/2, cy - 30 + h/2).
            # Target left-edge  MIDDLE point: (cx - w/2, cy - 30 + h/2).
            # center is FIXED 30px below the top edge (TRNSYS convention).
            # For 40x40 this reduces to +20 / -20 and cy-10 (no change).
            sw, sh = size_lookup.get(from_unit, (40, 40))
            tw, th = size_lookup.get(to_unit, (40, 40))
            start_x = from_pos["x"] + sw / 2  # Right edge
            start_y = from_pos["y"] - 30 + sh / 2  # middle slot
            end_x = to_pos["x"] - tw / 2  # Left edge
            end_y = to_pos["y"] - 30 + th / 2  # middle slot

            # Create path with intermediate point for orthogonal routing
            mid_x = _snap_to_grid((start_x + end_x) // 2)
            coordinates = [
                [start_x, start_y],
                [mid_x, start_y],
                [mid_x, end_y],
                [end_x, end_y],
            ]

            link_styles.append({
                "from_unit": from_unit,
                "to_unit": to_unit,
                "start_dir": f"{int(sw)}:{int(sh / 2)}",
                "end_dir": f"0:{int(th / 2)}",
                "input_indices": input_indices,
                "color": 0,
                "style": 0,
                "width": 2,
                "user_path": 0,
                "coordinates": coordinates,
            })

        return link_styles

    def _generate_link_style_lines_from_data(
        self,
        link_styles: List[Dict[str, Any]],
    ) -> List[Dict[str, str]]:
        """Generate LINK_STYLE line items from link_styles data.

        Each item is a dict with:
        - LINK: "from_unit:to_unit"
        - CONNECTION_SET: "start_dir:end_dir:indices:color:style:width:user_path:coordinates"

        Note: *!LINK_STYLE and *!LINK_STYLE_END are fixed DCK format wrappers
        and are NOT included here. They are added during final DCK assembly.
        """
        items = []

        for style in link_styles:
            from_unit = style.get("from_unit")
            to_unit = style.get("to_unit")

            if not from_unit or not to_unit:
                continue

            # Build CONNECTION_SET value
            start_dir = style.get("start_dir", "40:20")
            end_dir = style.get("end_dir", "0:20")
            input_indices = style.get("input_indices", [])
            color = style.get("color", 0)
            line_style = style.get("style", 0)
            width = style.get("width", 0)
            user_path = style.get("user_path", 0)  # Always 0
            coordinates = style.get("coordinates", [])

            # Format input indices
            if isinstance(input_indices, list):
                indices_str = ",".join(str(i) for i in input_indices)
            else:
                indices_str = str(input_indices)

            # Format coordinates
            coord_str = ":".join(f"{_fmt_coord(c[0])},{_fmt_coord(c[1])}" for c in coordinates)

            # Build CONNECTION_SET: start_dir:end_dir:indices:color:style:width:user_path:coordinates
            connection_set_value = f"{start_dir}:{end_dir}:{indices_str}:{color}:{line_style}:{width}:{user_path}:{coord_str}"

            items.append({
                "LINK": f"{from_unit}:{to_unit}",
                "CONNECTION_SET": connection_set_value,
            })

        return items

    def _post_process_vertical_segments(self, layout_data: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve vertical segment overlaps between adjacent unit columns.

        Processes column by column (left to right).  For each column gap:
        1. Extract vertical segments in the gap.
        2. Sort them by source unit position.
        3. Check whether the rightmost assigned X would be too close to the
           right column's left edge (< 40 px clearance).
        4. If so, shift ALL units and coordinates to the right of this gap
           rightward, then restart (do NOT assign X values yet).
        5. Once the gap is large enough, assign equally-spaced X values.

        This integrates the former two-pass approach (vertical overlap
        resolution + separate spacing enforcement) into a single pass that
        guarantees no vertical line overshoots the right column.
        """
        layout = layout_data.get("layout", [])
        link_styles = layout_data.get("link_styles", [])

        if not layout or not link_styles:
            return layout_data

        ICON_HALF = 20    # fallback half-width (40x40 icon)
        MAX_ITER = 30     # safety limit

        # Per-column icon half-width (variable icon sizes). Layout nodes carry
        # model_path, so we can resolve each column's real edge instead of 20.
        unit_size_map = build_unit_size_map(layout)
        col_half = build_column_half_widths(layout, unit_size_map)

        def _left_hw(col_x: int) -> float:
            return col_half.get(col_x, ICON_HALF)

        def _right_hw(col_x: int) -> float:
            return col_half.get(col_x, ICON_HALF)

        for _iteration in range(MAX_ITER):
            # (Re-)compute column X values after any shift
            unit_x_coords = sorted(
                set(int(u.get("position_x", 0)) for u in layout
                    if u.get("position_x") is not None)
            )
            if len(unit_x_coords) < 2:
                break

            shifted = False

            for col_idx in range(len(unit_x_coords) - 1):
                left_col_x = unit_x_coords[col_idx]
                right_col_x = unit_x_coords[col_idx + 1]

                # --- 1. Extract vertical segments in this gap ---
                v_segments = self._extract_vertical_segments(
                    link_styles, left_col_x, right_col_x
                )
                if not v_segments:
                    continue

                # --- 2. Sort ---
                sorted_segs = self._sort_vertical_segments(v_segments, layout, link_styles)
                n_segs = len(sorted_segs)
                if n_segs == 0:
                    continue

                # --- 3. Check if gap is large enough BEFORE assigning X ---
                # Use each column's ACTUAL icon half-width for its edges.
                base_x = _snap_to_grid(int(left_col_x + _left_hw(left_col_x) + FIRST_VLINE_GAP))
                rightmost_x = base_x + VLINE_INCREMENT * (n_segs - 1)
                right_left_edge = right_col_x - _right_hw(right_col_x)
                distance = right_left_edge - rightmost_x

                if distance < LAST_VLINE_GAP:
                    # Gap too small — shift right-side units FIRST
                    shift = LAST_VLINE_GAP - distance

                    for u in layout:
                        ux = int(u.get("position_x", 0))
                        if ux >= right_col_x:
                            u["position_x"] = ux + shift

                    # Threshold = right column's real left edge, so connection
                    # endpoints landing on that edge move WITH the unit (avoids
                    # the edge point detaching from a non-40px icon).
                    threshold_x = right_col_x - _right_hw(right_col_x)
                    for ls in link_styles:
                        coords = ls.get("coordinates", [])
                        for k in range(len(coords)):
                            if coords[k][0] >= threshold_x:
                                coords[k] = [coords[k][0] + shift, coords[k][1]]

                    shifted = True
                    break  # restart — do NOT assign X values yet

                # --- 4. Gap is large enough — assign equally-spaced X ---
                self._adjust_vertical_segment_x(
                    link_styles, sorted_segs, left_col_x, right_col_x,
                    left_half_width=_left_hw(left_col_x)
                )

            if not shifted:
                break

        layout_data["layout"] = layout
        layout_data["link_styles"] = link_styles
        return layout_data

    def _extract_vertical_segments(
        self,
        link_styles: List[Dict[str, Any]],
        interval_start: float,
        interval_end: float
    ) -> List[Dict[str, Any]]:
        """
        Extract vertical segments (middle waypoints) in the specified interval.

        For standard Z-shape paths (4 points), middle indices [1,2] share one X.
        For L+Z paths (6 points), there are TWO independent vertical segments
        with different X values — indices [1,2] (L-segment) and [3,4] (Z-segment).
        These are returned as separate entries so that _adjust_vertical_segment_x
        can assign each its own X without collapsing the L+Z structure.

        Returns list of dicts with:
        - link_style_index: index in link_styles
        - middle_indices: list of indices that share the SAME vertical X
        - from_unit: source unit number
        - to_unit: target unit number
        """
        segments = []

        for link_idx, link_style in enumerate(link_styles):
            coordinates = link_style.get("coordinates", [])

            if len(coordinates) < 4:
                continue

            from_u = link_style.get("from_unit")
            to_u = link_style.get("to_unit")

            # Group consecutive middle points by their X-coordinate.
            # Points with the same X belong to the same vertical segment.
            middle_indices = list(range(1, len(coordinates) - 1))

            groups: List[List[int]] = []
            current_group: List[int] = [middle_indices[0]]
            for mi in middle_indices[1:]:
                if coordinates[mi][0] == coordinates[current_group[0]][0]:
                    current_group.append(mi)
                else:
                    groups.append(current_group)
                    current_group = [mi]
            groups.append(current_group)

            # Each group is an independent vertical segment; check if it
            # falls within this interval.
            for grp in groups:
                wp_x = coordinates[grp[0]][0]
                if interval_start < wp_x < interval_end:
                    segments.append({
                        "link_style_index": link_idx,
                        "middle_indices": grp,
                        "from_unit": from_u,
                        "to_unit": to_u,
                    })

        return segments

    def _sort_vertical_segments(
        self,
        segments: List[Dict[str, Any]],
        layout: List[Dict[str, Any]],
        link_styles: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Sort vertical segments by their path type and start point coordinates.

        Sorting rule:
        1. Classify each segment as Z-type (4 points) or LZ-type (6+ points)
        2. Z-type segments are placed before LZ-type segments
        3. Within each type, sort by source unit center: x-coordinate (left first), then y-coordinate (top first)
        """
        # Build unit position map
        unit_positions = {}
        for unit in layout:
            unit_num = unit.get("unit_number")
            px = int(unit.get("position_x", 0))
            py = int(unit.get("position_y", 0))
            if unit_num is not None:
                unit_positions[unit_num] = (px, py)

        # Calculate start point and classify path type for each segment
        segments_with_start = []

        for segment in segments:
            from_unit = segment.get("from_unit")

            if from_unit not in unit_positions:
                continue

            unit_center = unit_positions[from_unit]

            # Classify path type by coordinate count
            link_idx = segment.get("link_style_index")
            coords = link_styles[link_idx].get("coordinates", []) if link_idx < len(link_styles) else []
            path_type = "LZ" if len(coords) >= 6 else "Z"

            segments_with_start.append({
                **segment,
                "unit_center": unit_center,
                "path_type": path_type,
            })

        # Sort by: path type (Z first), then unit center x (left first), then y (top first)
        sorted_segments = sorted(
            segments_with_start,
            key=lambda s: (
                0 if s.get("path_type") == "Z" else 1,
                s["unit_center"][0],
                s["unit_center"][1]
            )
        )

        return sorted_segments

    def _adjust_vertical_segment_x(
        self,
        link_styles: List[Dict[str, Any]],
        sorted_segments: List[Dict[str, Any]],
        interval_start: float,
        interval_end: float,
        left_half_width: float = 20,
    ) -> None:
        """
        Reassign vertical segment x-coordinates to guarantee no overlaps.

        Uses left unit right edge + FIRST_VLINE_GAP as the baseline, then assigns
        x = base_x + VLINE_INCREMENT * rank for each sorted segment.
        Adjusts ALL middle waypoints (supports both 4-point and 5-point paths).

        ``left_half_width`` is the left column's actual icon half-width (its right
        edge = interval_start + left_half_width); defaults to 20 (40x40 icon).
        """
        # Baseline: left unit right edge (interval_start + half_width) + FIRST_VLINE_GAP
        base_x = _snap_to_grid(int(interval_start + left_half_width + FIRST_VLINE_GAP))

        for rank, segment in enumerate(sorted_segments):
            link_idx = segment.get("link_style_index")
            middle_indices = segment.get("middle_indices", [])

            if link_idx >= len(link_styles):
                continue

            link_style = link_styles[link_idx]
            coordinates = link_style.get("coordinates", [])

            if len(coordinates) < 4:
                continue

            # Reassign x for ALL middle waypoints (not incremental, absolute)
            new_x = base_x + VLINE_INCREMENT * rank

            for mid_idx in middle_indices:
                if mid_idx < len(coordinates):
                    wp = coordinates[mid_idx]
                    coordinates[mid_idx] = [new_x, wp[1]]

            link_style["coordinates"] = coordinates

    # ------------------------------------------------------------------
    # Post-process: resolve horizontal segment overlaps in row gaps
    # ------------------------------------------------------------------

    def _post_process_horizontal_segments(
        self, layout_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Resolve horizontal segment overlaps between adjacent unit rows.

        Processes row by row (top to bottom).  For each row gap:
        1. Collect all horizontal segments whose source unit sits on the
           upper row (ownership rule).
        2. Sort them by source unit X (left-to-right).
        3. Assign equally-spaced Y values starting at upper_bottom_edge + 40.
        4. If the bottommost assigned Y is too close to the lower row's top
           edge (< 40 px clearance), shift ALL units and coordinates below
           that gap downward so that clearance == 40.  Then re-scan rows
           because positions have changed.

        This replaces the former two-pass approach (horizontal overlap
        resolution + separate row-spacing enforcement) with a single
        integrated pass that guarantees correctness.
        """
        layout = layout_data.get("layout", [])
        link_styles = layout_data.get("link_styles", [])

        if not layout or not link_styles:
            return layout_data

        ICON_BOTTOM_OFFSET = 10   # fallback cy -> bottom edge (40x40); per-row below
        ICON_TOP_OFFSET = 30      # cy - 30 = top edge (FIXED by TRNSYS convention)
        MAX_ITER = 30             # safety limit (shifts may cascade)

        # Per-row icon edge offsets (variable icon sizes). top offset is always 30;
        # bottom offset = max(height - 30) over the row's icons.
        unit_size_map = build_unit_size_map(layout)
        row_off = build_row_edge_offsets(layout, unit_size_map)

        def _bottom_off(row_y: int) -> float:
            return row_off.get(row_y, (30.0, float(ICON_BOTTOM_OFFSET)))[1]

        def _top_off(row_y: int) -> float:
            # Top edge is always 30 below... above the centre (TRNSYS convention).
            return row_off.get(row_y, (float(ICON_TOP_OFFSET), 10.0))[0]

        for _iteration in range(MAX_ITER):
            # (Re-)compute row Y values after any shift
            unit_y_coords = sorted(
                set(int(u.get("position_y", 0)) for u in layout
                    if u.get("position_y") is not None)
            )
            if len(unit_y_coords) < 2:
                break

            # Build unit position lookup
            unit_positions_map: Dict[int, Tuple[int, int]] = {}
            for unit in layout:
                uid = unit.get("unit_number")
                if uid is not None:
                    unit_positions_map[uid] = (
                        int(unit.get("position_x", 0)),
                        int(unit.get("position_y", 0)),
                    )

            shifted = False

            for row_idx in range(len(unit_y_coords) - 1):
                upper_row_y = unit_y_coords[row_idx]
                lower_row_y = unit_y_coords[row_idx + 1]

                # --- 1. Extract horizontal segments owned by this row ---
                # Pass the row's ACTUAL edge offsets (variable icon sizes) so the
                # gap-membership test uses the real bottom/top edges, not 40x40.
                h_segments = self._extract_horizontal_segments(
                    link_styles, upper_row_y, lower_row_y,
                    unit_positions=unit_positions_map,
                    upper_bottom_off=_bottom_off(upper_row_y),
                    lower_top_off=_top_off(lower_row_y),
                )
                if not h_segments:
                    continue

                # --- 2. Sort by source unit X (left first) ---
                sorted_segs = self._sort_horizontal_segments(h_segments, layout)
                n_segs = len(sorted_segs)
                if n_segs == 0:
                    continue

                # --- 3. Check if gap is large enough BEFORE assigning Y ---
                # NOTE: Do NOT snap to grid=20. Unit cy is on 35+16n grid, so
                # bottom_edge (cy+bottom_off) + FIRST_HLINE_GAP yields a value
                # already on 16-grid. Snapping to 20-grid would produce row-
                # dependent offsets, breaking the gap=FIRST_HLINE_GAP guarantee.
                # Use the upper row's ACTUAL bottom edge and the lower row's
                # top edge (always 30).
                base_y = int(upper_row_y + _bottom_off(upper_row_y) + FIRST_HLINE_GAP)
                bottommost_y = base_y + HLINE_INCREMENT * (n_segs - 1)
                lower_top_edge = lower_row_y - _top_off(lower_row_y)
                distance = lower_top_edge - bottommost_y

                if distance < LAST_HLINE_GAP:
                    # Gap too small — shift lower units down FIRST,
                    # then restart so Y assignment uses the enlarged gap.
                    # Round shift up to the nearest multiple of 16 so that
                    # unit centres stay on the y = 35 + 16n grid after shifting.
                    shift = ((LAST_HLINE_GAP - distance + 15) // 16) * 16

                    for u in layout:
                        uy = int(u.get("position_y", 0))
                        if uy >= lower_row_y:
                            u["position_y"] = uy + shift

                    # Threshold = lower row's top edge (always cy - 30), so
                    # connection endpoints on that edge move WITH the unit.
                    threshold_y = lower_row_y - _top_off(lower_row_y)
                    for ls in link_styles:
                        coords = ls.get("coordinates", [])
                        for k in range(len(coords)):
                            if coords[k][1] >= threshold_y:
                                coords[k] = [coords[k][0], coords[k][1] + shift]

                    shifted = True
                    break  # restart — do NOT assign Y values yet

                # --- 4. Gap is large enough — assign equally-spaced Y ---
                for rank, segment in enumerate(sorted_segs):
                    link_idx = segment.get("link_style_index")
                    middle_indices = segment.get("middle_indices", [])
                    if link_idx >= len(link_styles):
                        continue
                    link_style = link_styles[link_idx]
                    coordinates = link_style.get("coordinates", [])
                    if len(coordinates) < 5:
                        continue
                    new_y = base_y + HLINE_INCREMENT * rank
                    for mid_idx in middle_indices:
                        if mid_idx < len(coordinates):
                            wp = coordinates[mid_idx]
                            coordinates[mid_idx] = [wp[0], new_y]
                    link_style["coordinates"] = coordinates

            if not shifted:
                break

        layout_data["layout"] = layout
        layout_data["link_styles"] = link_styles
        return layout_data

    def _extract_horizontal_segments(
        self,
        link_styles: List[Dict[str, Any]],
        interval_start: float,
        interval_end: float,
        unit_positions: Optional[Dict[int, Tuple[int, int]]] = None,
        upper_bottom_off: float = 10,
        lower_top_off: float = 30,
    ) -> List[Dict[str, Any]]:
        """Extract horizontal segments in the specified row interval.

        Only processes paths with >= 5 points (L+Z routing).
        Groups consecutive middle points by Y-coordinate; a group whose Y
        falls strictly between the upper unit's bottom edge and the lower
        unit's top edge is returned as one horizontal segment entry.

        Ownership rule: a horizontal segment belongs to a row gap only if
        its source unit (from_unit) sits on the upper row of that gap
        (i.e. from_unit's centre_y == interval_start).  This ensures
        horizontal lines travel through the gap immediately below their
        source unit.

        Icon geometry: centre (cx, cy), top edge cy-30 (fixed), bottom edge
        cy + (height-30). ``upper_bottom_off`` / ``lower_top_off`` carry the
        ACTUAL per-row offsets (default 10/30 = 40x40 icon).
        """
        upper_bottom_edge = interval_start + upper_bottom_off
        lower_top_edge = interval_end - lower_top_off

        segments: List[Dict[str, Any]] = []

        for link_idx, link_style in enumerate(link_styles):
            coordinates = link_style.get("coordinates", [])
            if len(coordinates) < 5:
                continue

            from_u = link_style.get("from_unit")
            to_u = link_style.get("to_unit")

            # Ownership check: only include if source unit is on the upper row
            if unit_positions and from_u is not None:
                src_cy = unit_positions.get(from_u, (0, 0))[1]
                if src_cy != interval_start:
                    continue

            middle_indices = list(range(1, len(coordinates) - 1))

            # Group consecutive middle points by Y
            groups: List[List[int]] = []
            current_group: List[int] = [middle_indices[0]]
            for mi in middle_indices[1:]:
                if coordinates[mi][1] == coordinates[current_group[0]][1]:
                    current_group.append(mi)
                else:
                    groups.append(current_group)
                    current_group = [mi]
            groups.append(current_group)

            for grp in groups:
                wp_y = coordinates[grp[0]][1]
                if upper_bottom_edge < wp_y < lower_top_edge:
                    segments.append({
                        "link_style_index": link_idx,
                        "middle_indices": grp,
                        "from_unit": from_u,
                        "to_unit": to_u,
                    })

        return segments

    def _sort_horizontal_segments(
        self,
        segments: List[Dict[str, Any]],
        layout: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Sort horizontal segments by their source unit position.

        Sorting rule: y-coordinate first (top first), then x (left first).
        """
        unit_positions: Dict[int, Tuple[int, int]] = {}
        for unit in layout:
            unit_num = unit.get("unit_number")
            px = int(unit.get("position_x", 0))
            py = int(unit.get("position_y", 0))
            if unit_num is not None:
                unit_positions[unit_num] = (px, py)

        segments_with_start = []
        for segment in segments:
            from_unit = segment.get("from_unit")
            if from_unit not in unit_positions:
                continue
            unit_center = unit_positions[from_unit]
            segments_with_start.append({
                **segment,
                "unit_center": unit_center,
            })

        return sorted(
            segments_with_start,
            key=lambda s: (s["unit_center"][1], s["unit_center"][0]),
        )

    def _merge_connection_segments(
        self,
        layout_data: Dict[str, Any],
        components: List[Dict[str, Any]],
        equations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Merge overlapping connection segments based on connection patterns.

        This post-processing step reduces visual clutter by merging connection
        segments that follow the same path. Applies 8 rules to identify and
        merge segments for connections between components of the same type.

        Args:
            layout_data: Contains 'layout' and 'link_styles' lists
            components: List of component dicts with unit_number, type_number, etc.
            equations: List of equation dicts with unit_number, unit_name, etc.

        Returns:
            Updated layout_data with merged connection segments

        Rules:
        - Rule 1: Same type, multiple sources → single target (adjacent columns, Z-path)
                  Merge segments 2 and 3 (vertical + horizontal)
        - Rule 2: Single source → same type, multiple targets (adjacent columns, Z-path)
                  Merge segments 1 and 2 (horizontal + vertical)
        - Rule 3: Same type, multiple sources → single target (multi-column, L+Z path)
                  Merge L's segment 2 + all Z segments
        - Rule 4: Single source → same type, multiple targets (multi-column, L+Z path)
                  Merge all L segments + Z's segments 1 and 2
        - Rules 5-8: Same as 1-4, but treat all formulas in same column as same type

        Only processes when enable_connection_segment_merging=1 in performance_config.
        """
        from collections import defaultdict
        from ...core.performance_config import get_performance_config

        # Check config switch
        perf_config = get_performance_config()
        if perf_config.enable_connection_segment_merging == 0:
            return layout_data

        link_styles = layout_data.get("link_styles", [])
        layout = layout_data.get("layout", [])

        if not link_styles or not layout:
            return layout_data

        # Build type mapping: unit_number -> type_number (for components)
        unit_type_map = {}
        for comp in components:
            unit_num = comp.get("unit_number")
            type_num = comp.get("type_number")
            if unit_num and type_num:
                unit_type_map[unit_num] = type_num

        # Build formula set: unit_numbers that are formulas
        formula_units = set()
        for eq in equations:
            unit_num = eq.get("unit_number")
            if unit_num:
                formula_units.add(unit_num)

        # Build position mapping: unit_number -> (x, y)
        unit_position_map = {}
        for item in layout:
            unit_num = item.get("unit_number")
            x = item.get("position_x")
            y = item.get("position_y")
            if unit_num and x is not None and y is not None:
                unit_position_map[unit_num] = (x, y)

        # Build icon size mapping: unit_number -> (width, height)
        # Required so post-merge direction codes reflect the actual icon size.
        # IMPORTANT: read model_path from the AUTHORITATIVE components/equations
        # lists, NOT from layout nodes (LLM layout nodes have no model_path, which
        # would make every icon default to 40x40).
        unit_model_path = {}
        for comp in components:
            un = comp.get("unit_number")
            if un is not None:
                unit_model_path[un] = comp.get("model_path", "") or ""
        unit_size_map = {}
        for item in layout:
            unit_num = item.get("unit_number")
            if unit_num is not None:
                is_equation = unit_num in formula_units
                model_path = unit_model_path.get(unit_num, "")
                component_id = extract_component_id(model_path)
                unit_size_map[unit_num] = get_icon_size(component_id, is_equation)

        # Helper: Get type identifier for a unit
        def get_unit_type(unit_num: int) -> Optional[str]:
            """Get type identifier: type_number for components, 'FORMULA' for equations"""
            if unit_num in formula_units:
                return "FORMULA"
            return unit_type_map.get(unit_num)

        # Helper: Check if two units are the same type
        def is_same_type(unit1: int, unit2: int) -> bool:
            """Check if two units have the same type"""
            type1 = get_unit_type(unit1)
            type2 = get_unit_type(unit2)
            if type1 is None or type2 is None:
                return False
            # Formulas in same column are considered same type
            if type1 == "FORMULA" and type2 == "FORMULA":
                pos1 = unit_position_map.get(unit1)
                pos2 = unit_position_map.get(unit2)
                if pos1 and pos2:
                    return pos1[0] == pos2[0]  # Same column
            return type1 == type2

        # Helper: Check if connection is adjacent columns or multi-column
        def is_adjacent_columns(from_unit: int, to_unit: int) -> bool:
            """Check if connection spans only adjacent columns"""
            from_pos = unit_position_map.get(from_unit)
            to_pos = unit_position_map.get(to_unit)
            if not from_pos or not to_pos:
                return True  # Default to adjacent
            x_distance = abs(from_pos[0] - to_pos[0])
            return x_distance <= LAYOUT_MIN_COL_SPACING

        # Helper: Compute direction code based on unit center and connection point
        def compute_direction_code(unit_center_x: int, unit_center_y: int,
                                   point_x: int, point_y: int,
                                   icon_width: int = 40, icon_height: int = 40) -> str:
            """
            Compute direction code (format: X:Y) for a connection point.

            The direction code is the connection point expressed in the icon's
            local coordinate system (origin = icon top-left):
              - X: left edge -> 0, right edge -> icon_width
              - Y: top -> 0, middle -> icon_height/2, bottom -> icon_height

            This mirrors the variable-size mapping used in
            _compute_connection_coordinates Step 3b+, so the codes stay
            consistent with the actual icon size instead of assuming 40x40.

            Args:
                unit_center_x, unit_center_y: Unit center coordinates
                point_x, point_y: Connection point coordinates
                icon_width, icon_height: Actual icon size (default 40x40)

            Returns:
                Direction code string like "37:0" or "0:50"
            """
            # X code: based on horizontal position relative to center
            if point_x < unit_center_x:
                x_code = 0  # Left edge
            elif point_x > unit_center_x:
                x_code = icon_width  # Right edge (actual width)
            else:
                x_code = int(icon_width / 2)  # Center (shouldn't happen for edge connections)

            # Y code: map the vertical offset to {0, height/2, height} using the
            # same offset reference as Step 3b+ (get_edge_offsets).
            y_offsets, _ = get_edge_offsets(icon_width, icon_height)
            y_offset = point_y - unit_center_y
            if abs(y_offset - y_offsets[0]) < abs(y_offset - y_offsets[1]):
                y_code = 0  # Top
            elif abs(y_offset - y_offsets[1]) < abs(y_offset - y_offsets[2]):
                y_code = int(icon_height / 2)  # Middle (actual height)
            else:
                y_code = icon_height  # Bottom (actual height)

            return f"{x_code}:{y_code}"

        # Group connections by pattern for Rules 1, 3, 5, 7
        # Pattern: Multiple same-type sources → single target
        target_to_sources = defaultdict(list)
        for link in link_styles:
            from_unit = link.get("from_unit")
            to_unit = link.get("to_unit")
            if from_unit and to_unit:
                target_to_sources[to_unit].append(link)

        # Group connections by pattern for Rules 2, 4, 6, 8
        # Pattern: Single source → multiple same-type targets
        source_to_targets = defaultdict(list)
        for link in link_styles:
            from_unit = link.get("from_unit")
            to_unit = link.get("to_unit")
            if from_unit and to_unit:
                source_to_targets[from_unit].append(link)

        # Process Rules 1, 3, 5, 7: Multiple same-type sources → single target
        for target_unit, links in target_to_sources.items():
            if len(links) < 2:
                continue

            # Group by source unit type
            type_groups = defaultdict(list)
            for link in links:
                from_unit = link.get("from_unit")
                unit_type = get_unit_type(from_unit)
                if unit_type:
                    type_groups[unit_type].append(link)

            # Process each type group (at least 2 connections of same type)
            for unit_type, type_links in type_groups.items():
                if len(type_links) < 2:
                    continue

                # Use first link as reference
                ref_link = type_links[0]
                ref_coords = ref_link.get("coordinates", [])

                # Determine path type by coordinate count, not by column distance
                # This ensures we handle the actual path structure correctly
                num_points = len(ref_coords)

                if num_points == 4:
                    # Z-path (4 points): merge segments 2 and 3
                    # Path structure: [start, waypoint1, waypoint2, end]
                    ref_wp1_x = ref_coords[1][0]
                    ref_wp2 = ref_coords[2]
                    ref_end = ref_coords[3]

                    # Update all connections in this type group
                    for link in type_links:
                        coords = link.get("coordinates", [])
                        to_unit = link.get("to_unit")
                        if len(coords) == 4:
                            # Update waypoint1: keep y, use ref x
                            coords[1] = [ref_wp1_x, coords[1][1]]
                            # Update waypoint2: use ref waypoint2
                            coords[2] = [ref_wp2[0], ref_wp2[1]]
                            # Update end: use ref end to avoid diagonal lines
                            coords[3] = [ref_end[0], ref_end[1]]
                            link["coordinates"] = coords

                            # Update end_dir
                            to_pos = unit_position_map.get(to_unit)
                            if to_pos:
                                tw, th = unit_size_map.get(to_unit, (40, 40))
                                link["end_dir"] = compute_direction_code(
                                    to_pos[0], to_pos[1],
                                    coords[-1][0], coords[-1][1],
                                    tw, th
                                )

                elif num_points == 6:
                    # L+Z path (6 points): merge L's segment 2 + all Z segments
                    # Path structure: [start, L1, L2, Z1, Z2, end]
                    ref_L1_x = ref_coords[1][0]
                    ref_L2 = ref_coords[2]
                    ref_Z1 = ref_coords[3]
                    ref_Z2 = ref_coords[4]
                    ref_end = ref_coords[5]

                    # Update all connections in this type group
                    for link in type_links:
                        coords = link.get("coordinates", [])
                        to_unit = link.get("to_unit")
                        if len(coords) == 6:
                            # Update L1: keep y, use ref x
                            coords[1] = [ref_L1_x, coords[1][1]]
                            # Update L2: use ref L2
                            coords[2] = [ref_L2[0], ref_L2[1]]
                            # Update Z1: use ref Z1
                            coords[3] = [ref_Z1[0], ref_Z1[1]]
                            # Update Z2: use ref Z2
                            coords[4] = [ref_Z2[0], ref_Z2[1]]
                            # Update end: use ref end to avoid diagonal lines
                            coords[5] = [ref_end[0], ref_end[1]]
                            link["coordinates"] = coords

                            # Update end_dir
                            to_pos = unit_position_map.get(to_unit)
                            if to_pos:
                                tw, th = unit_size_map.get(to_unit, (40, 40))
                                link["end_dir"] = compute_direction_code(
                                    to_pos[0], to_pos[1],
                                    coords[-1][0], coords[-1][1],
                                    tw, th
                                )

        # Process Rules 2, 4, 6, 8: Single source → multiple same-type targets
        for source_unit, links in source_to_targets.items():
            if len(links) < 2:
                continue

            # Group by target unit type
            type_groups = defaultdict(list)
            for link in links:
                to_unit = link.get("to_unit")
                unit_type = get_unit_type(to_unit)
                if unit_type:
                    type_groups[unit_type].append(link)

            # Process each type group (at least 2 connections of same type)
            for unit_type, type_links in type_groups.items():
                if len(type_links) < 2:
                    continue

                # Use first link as reference
                ref_link = type_links[0]
                ref_coords = ref_link.get("coordinates", [])

                # Determine path type by coordinate count, not by column distance
                # This ensures we handle the actual path structure correctly
                num_points = len(ref_coords)

                if num_points == 4:
                    # Z-path (4 points): merge segments 1 and 2
                    # Path structure: [start, waypoint1, waypoint2, end]
                    ref_start_y = ref_coords[0][1]
                    ref_wp1 = ref_coords[1]
                    ref_wp2_x = ref_coords[2][0]

                    # Update all connections in this type group
                    for link in type_links:
                        coords = link.get("coordinates", [])
                        from_unit = link.get("from_unit")
                        if len(coords) == 4:
                            # Update start: keep x, use ref y
                            coords[0] = [coords[0][0], ref_start_y]
                            # Update waypoint1: use ref waypoint1
                            coords[1] = [ref_wp1[0], ref_wp1[1]]
                            # Update waypoint2: use ref x, keep y
                            coords[2] = [ref_wp2_x, coords[2][1]]
                            link["coordinates"] = coords

                            # Update start_dir
                            from_pos = unit_position_map.get(from_unit)
                            if from_pos:
                                fw, fh = unit_size_map.get(from_unit, (40, 40))
                                link["start_dir"] = compute_direction_code(
                                    from_pos[0], from_pos[1],
                                    coords[0][0], coords[0][1],
                                    fw, fh
                                )

                elif num_points == 6:
                    # L+Z path (6 points): merge all L segments + Z's segments 1 and 2
                    # Path structure: [start, L1, L2, Z1, Z2, end]
                    ref_start_y = ref_coords[0][1]
                    ref_L1 = ref_coords[1]
                    ref_L2 = ref_coords[2]
                    ref_Z1 = ref_coords[3]
                    ref_Z2_x = ref_coords[4][0]

                    # Update all connections in this type group
                    for link in type_links:
                        coords = link.get("coordinates", [])
                        from_unit = link.get("from_unit")
                        if len(coords) == 6:
                            # Update start: keep x, use ref y
                            coords[0] = [coords[0][0], ref_start_y]
                            # Update L1: use ref L1
                            coords[1] = [ref_L1[0], ref_L1[1]]
                            # Update L2: use ref L2
                            coords[2] = [ref_L2[0], ref_L2[1]]
                            # Update Z1: use ref Z1
                            coords[3] = [ref_Z1[0], ref_Z1[1]]
                            # Update Z2: use ref x, keep y
                            coords[4] = [ref_Z2_x, coords[4][1]]
                            link["coordinates"] = coords

                            # Update start_dir
                            from_pos = unit_position_map.get(from_unit)
                            if from_pos:
                                fw, fh = unit_size_map.get(from_unit, (40, 40))
                                link["start_dir"] = compute_direction_code(
                                    from_pos[0], from_pos[1],
                                    coords[0][0], coords[0][1],
                                    fw, fh
                                )

        # Post-merge enhancement: reassign edge points and readjust spacing
        layout_data = self._reassign_edge_points_after_merge(
            layout_data, unit_position_map, compute_direction_code, unit_size_map
        )
        layout_data = self._readjust_spacing_after_merge(layout_data)

        layout_data["link_styles"] = link_styles
        return layout_data

    def _reassign_edge_points_after_merge(
        self,
        layout_data: Dict[str, Any],
        unit_position_map: Dict[int, Tuple[int, int]],
        compute_direction_code: callable,
        unit_size_map: Optional[Dict[int, Tuple[int, int]]] = None
    ) -> Dict[str, Any]:
        """
        Reassign edge connection points after segment merging.

        Post-merge edge points are assigned by individual link type:

        - L+Z connections (6 coordinate points, used for non-adjacent columns):
          all links are pinned to the BOTTOM slot (cy + (height - 30)).
        - Z connections (4 coordinate points, used for adjacent columns):
          links alternate between CENTER and TOP slots
          (CENTER = cy - 30 + height/2, TOP = cy - 30),
          ranked by original Y coordinate then link index.

        Slot Y offsets follow the ACTUAL icon size (TRNSYS convention: center is
        a FIXED 30px below the top edge). For a 40x40 icon these reduce to the
        classic +10 / -10 / -30 (no regression).

        Each link receives its own Y coordinate independently. When multiple
        Z links share the same physical point, they are distributed across
        center/top slots (1st→center, 2nd→top, 3rd→center, 4th→top, ...).

        Args:
            layout_data: Contains 'layout' and 'link_styles' lists
            unit_position_map: Mapping from unit_number to (x, y) position
            compute_direction_code: Function to compute direction code
            unit_size_map: Mapping from unit_number to (width, height) icon size.
                Used so direction codes reflect the actual icon size. Defaults
                to 40x40 for any missing unit.

        Returns:
            Updated layout_data with reassigned edge points
        """
        from collections import defaultdict

        if unit_size_map is None:
            unit_size_map = {}

        link_styles = layout_data.get("link_styles", [])
        if not link_styles:
            return layout_data

        # Step 1: Group connections by their start/end point coordinates
        # Key: (unit, edge_side, x, y), Value: list of link indices
        start_point_groups = defaultdict(list)
        end_point_groups = defaultdict(list)

        for idx, link in enumerate(link_styles):
            from_unit = link.get("from_unit")
            to_unit = link.get("to_unit")
            coords = link.get("coordinates", [])

            if len(coords) < 2:
                continue

            # Group by start point
            if from_unit and from_unit in unit_position_map:
                from_pos = unit_position_map[from_unit]
                start_x, start_y = coords[0][0], coords[0][1]

                # Determine edge side
                if start_x < from_pos[0]:
                    from_edge = "left"
                elif start_x > from_pos[0]:
                    from_edge = "right"
                else:
                    from_edge = "left"  # same column, default to left

                start_key = (from_unit, from_edge, start_x, start_y)
                start_point_groups[start_key].append(idx)

            # Group by end point
            if to_unit and to_unit in unit_position_map:
                to_pos = unit_position_map[to_unit]
                end_x, end_y = coords[-1][0], coords[-1][1]

                # Determine edge side
                if end_x < to_pos[0]:
                    to_edge = "left"
                elif end_x > to_pos[0]:
                    to_edge = "right"
                else:
                    to_edge = "left"

                end_key = (to_unit, to_edge, end_x, end_y)
                end_point_groups[end_key].append(idx)

        # Step 2: For each unit edge, classify links directly by type (LZ or Z).
        # Each link is stored with its physical coordinates for later sorting.
        # Key: (unit, edge_side), Value: {"LZ": [(link_idx, x, y), ...], "Z": [...]}
        unit_edge_links: Dict[Tuple[int, str], Dict[str, List[Tuple[int, int, int, str]]]] = defaultdict(lambda: {"LZ": [], "Z": []})

        def _classify_links(unit, edge, x, y, link_indices, point_type):
            """Classify each link by type (LZ or Z) directly, with point type marker."""
            links_dict = unit_edge_links[(unit, edge)]
            for li in link_indices:
                if len(link_styles[li].get("coordinates", [])) >= 6:
                    links_dict["LZ"].append((li, x, y, point_type))
                else:
                    links_dict["Z"].append((li, x, y, point_type))

        for (unit, edge, x, y), indices in start_point_groups.items():
            _classify_links(unit, edge, x, y, indices, "start")

        for (unit, edge, x, y), indices in end_point_groups.items():
            _classify_links(unit, edge, x, y, indices, "end")

        # Step 3: For each unit edge, redistribute Y-coordinates by link type.
        # Slot Y follows the ACTUAL icon size (40x40 values shown in parens):
        # - All L+Z links → bottom slot (cy + (h - 30), i.e. +10 for 40x40)
        # - Z links → assign based on link direction to avoid crossings:
        #     * Start point: if target Y > source Y (downward link) → center
        #                    (cy - 30 + h/2, i.e. -10) else (upward/level) → top (cy - 30)
        #     * End point:   if source Y > target Y (link from above) → center
        #                    (cy - 30 + h/2) else (link from below/level) → top (cy - 30)
        # Key: (unit, edge_side, old_x, old_y, link_idx), Value: new_y
        new_y_map = {}

        for (unit, edge), links_dict in unit_edge_links.items():
            cy = unit_position_map.get(unit, (0, 0))[1]
            # Real icon size so the slot Y offsets follow the actual icon, not 40x40.
            # Edge-point Y rule (TRNSYS convention, center is FIXED 30 below top):
            #   top slot    = cy - 30                  (top edge, height-independent)
            #   center slot = cy - 30 + h/2            (middle edge)
            #   bottom slot = cy + (h - 30)            (bottom edge)
            # For 40x40 these are cy-30 / cy-10 / cy+10 (no change).
            _w, _h = unit_size_map.get(unit, (40, 40))
            bottom_y = cy + (_h - 30)
            center_y = cy - 30 + _h / 2
            top_y = cy - 30

            # All L+Z links assigned to bottom slot
            for (li, px, py, point_type) in links_dict["LZ"]:
                new_y_map[(unit, edge, px, py, li)] = bottom_y

            # Z links: assign based on link direction
            for (li, px, py, point_type) in links_dict["Z"]:
                link = link_styles[li]
                from_unit = link.get("from_unit")
                to_unit = link.get("to_unit")

                from_y = unit_position_map.get(from_unit, (0, 0))[1]
                to_y = unit_position_map.get(to_unit, (0, 0))[1]

                if point_type == "start":
                    # Start point: downward link → center, upward/level link → top
                    if to_y > from_y:
                        new_y = center_y  # center slot
                    else:
                        new_y = top_y  # top slot
                else:  # point_type == "end"
                    # End point: link from above → center, link from below/level → top
                    if from_y > to_y:
                        new_y = center_y  # center slot
                    else:
                        new_y = top_y  # top slot

                new_y_map[(unit, edge, px, py, li)] = new_y

        # Step 4: Update link_styles with new Y-coordinates (per-link lookup)
        for (unit, edge, old_x, old_y), link_indices in start_point_groups.items():
            for idx in link_indices:
                new_y = new_y_map.get((unit, edge, old_x, old_y, idx))
                if new_y is None:
                    continue

                link = link_styles[idx]
                coords = link.get("coordinates", [])

                if len(coords) >= 2:
                    # Update start point Y
                    coords[0] = [coords[0][0], new_y]

                    # Update adjacent segment endpoint to maintain line integrity
                    if len(coords) >= 6:
                        # L+Z connection: coords[0]→[1] is L vertical segment
                        # Must update coords[1] Y to keep the vertical line straight
                        coords[1] = [coords[1][0], new_y]
                    elif len(coords) >= 2 and coords[1][0] != coords[0][0]:
                        # Z connection: coords[0]→[1] is horizontal segment
                        # Update coords[1] Y to keep the horizontal line straight
                        coords[1] = [coords[1][0], new_y]

                    # Update start_dir
                    from_unit = link.get("from_unit")
                    from_pos = unit_position_map.get(from_unit)
                    if from_pos:
                        fw, fh = unit_size_map.get(from_unit, (40, 40))
                        link["start_dir"] = compute_direction_code(
                            from_pos[0], from_pos[1],
                            coords[0][0], coords[0][1],
                            fw, fh
                        )

        for (unit, edge, old_x, old_y), link_indices in end_point_groups.items():
            for idx in link_indices:
                new_y = new_y_map.get((unit, edge, old_x, old_y, idx))
                if new_y is None:
                    continue

                link = link_styles[idx]
                coords = link.get("coordinates", [])

                if len(coords) >= 2:
                    # Update end point Y
                    coords[-1] = [coords[-1][0], new_y]

                    # Update adjacent segment endpoint to maintain line integrity
                    if len(coords) >= 6:
                        # L+Z connection: coords[-2]→[-1] is Z vertical segment
                        # Must update coords[-2] Y to keep the vertical line straight
                        coords[-2] = [coords[-2][0], new_y]
                    elif len(coords) >= 2 and coords[-2][0] != coords[-1][0]:
                        # Z connection: coords[-2]→[-1] is horizontal segment
                        # Update coords[-2] Y to keep the horizontal line straight
                        coords[-2] = [coords[-2][0], new_y]

                    # Update end_dir
                    to_unit = link.get("to_unit")
                    to_pos = unit_position_map.get(to_unit)
                    if to_pos:
                        tw, th = unit_size_map.get(to_unit, (40, 40))
                        link["end_dir"] = compute_direction_code(
                            to_pos[0], to_pos[1],
                            coords[-1][0], coords[-1][1],
                            tw, th
                        )

        layout_data["link_styles"] = link_styles
        return layout_data

    def _readjust_spacing_after_merge(
        self,
        layout_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Readjust line spacing after segment merging.

        Strategy: This is essentially "re-run pre-processing with merged segments
        treated as ONE segment". Steps:
        1. Reset layout to compact state (LAYOUT_MIN_COL_SPACING / LAYOUT_MIN_ROW_SPACING)
        2. Run pre-processing logic (mirror of _post_process_vertical_segments and
           _post_process_horizontal_segments), with the only difference being that
           segments at the same X (or Y) coordinate are merged into one group.

        Key principle (matches pre-processing exactly):
        - Only ENLARGE gap, never shrink it (shift is always positive).
        - Use distance < LAST_VLINE_GAP / LAST_HLINE_GAP as the trigger.
        - Use n_groups (deduplicated count) instead of n_segs.

        After step 1, the layout is compact (small spacing). Then step 2 enlarges
        the spacing based on merged segment count, which is smaller than the
        original count, resulting in tighter (but correct) final spacing.

        Args:
            layout_data: Contains 'layout' and 'link_styles' lists

        Returns:
            Updated layout_data with readjusted spacing
        """
        from collections import defaultdict

        layout = layout_data.get("layout", [])
        link_styles = layout_data.get("link_styles", [])

        if not layout or not link_styles:
            return layout_data

        ICON_HALF = 20
        ICON_BOTTOM_OFFSET = 10
        ICON_TOP_OFFSET = 30
        MAX_ITER = 30  # safety limit for restart loop

        # Per-column/row icon edges for variable icon sizes (mirror of the
        # pre-processing passes). Layout nodes carry model_path.
        unit_size_map = build_unit_size_map(layout)
        col_half = build_column_half_widths(layout, unit_size_map)
        row_off = build_row_edge_offsets(layout, unit_size_map)

        def _col_hw(col_x: int) -> float:
            return col_half.get(col_x, ICON_HALF)

        def _bottom_off(row_y: int) -> float:
            return row_off.get(row_y, (30.0, float(ICON_BOTTOM_OFFSET)))[1]

        def _top_off(row_y: int) -> float:
            return row_off.get(row_y, (float(ICON_TOP_OFFSET), 10.0))[0]


        # =====================================================================
        # Step 1: Process vertical segments (column gaps) — ENLARGE pass
        # MIRROR of _post_process_vertical_segments. Only difference:
        # segments at the same X are merged into one group (n_groups vs n_segs).
        # This pass only ENLARGES gaps that are too small for the merged segments.
        # =====================================================================
        for _iteration in range(MAX_ITER):
            # (Re-)compute column X values after any shift
            unit_x_coords = sorted(
                set(int(u.get("position_x", 0)) for u in layout
                    if u.get("position_x") is not None)
            )
            if len(unit_x_coords) < 2:
                break

            shifted = False

            for col_idx in range(len(unit_x_coords) - 1):
                left_col_x = unit_x_coords[col_idx]
                right_col_x = unit_x_coords[col_idx + 1]

                # --- 1. Extract vertical segments in this gap ---
                v_segments = []
                for link_idx, link in enumerate(link_styles):
                    coords = link.get("coordinates", [])
                    for seg_idx in range(len(coords) - 1):
                        if coords[seg_idx][0] == coords[seg_idx + 1][0]:
                            x = coords[seg_idx][0]
                            if left_col_x + _col_hw(left_col_x) < x < right_col_x - _col_hw(right_col_x):
                                v_segments.append({
                                    "link_idx": link_idx,
                                    "seg_idx": seg_idx,
                                    "x": x,
                                })

                if not v_segments:
                    continue

                # --- 2. Group by X coordinate (CORE DIFFERENCE from pre-processing) ---
                # Merged segments share the same X — treat them as ONE group.
                x_groups = defaultdict(list)
                for seg in v_segments:
                    x_groups[seg["x"]].append(seg)

                n_groups = len(x_groups)
                if n_groups == 0:
                    continue

                # --- 2b. Classify each X group by path type (Z vs LZ) ---
                # If any segment in the group is LZ-type (>= 6 points), the whole group is LZ.
                x_types = {}
                for x, segs in x_groups.items():
                    group_type = "Z"
                    for seg in segs:
                        link_idx = seg["link_idx"]
                        coords = link_styles[link_idx].get("coordinates", [])
                        if len(coords) >= 6:
                            group_type = "LZ"
                            break
                    x_types[x] = group_type

                # --- 3. Check if gap is large enough BEFORE assigning X ---
                # (Identical to pre-processing, just using n_groups instead of n_segs)
                # Use each column's ACTUAL icon half-width for its edges.
                base_x = _snap_to_grid(int(left_col_x + _col_hw(left_col_x) + FIRST_VLINE_GAP))
                rightmost_x = base_x + VLINE_INCREMENT * (n_groups - 1)
                right_left_edge = right_col_x - _col_hw(right_col_x)
                distance = right_left_edge - rightmost_x

                if distance < LAST_VLINE_GAP:
                    # Gap too small — shift right-side units FIRST
                    shift = LAST_VLINE_GAP - distance

                    for u in layout:
                        ux = int(u.get("position_x", 0))
                        if ux >= right_col_x:
                            u["position_x"] = ux + shift

                    # Threshold = right column's real left edge so endpoints on
                    # that edge move WITH the unit (variable icon sizes).
                    threshold_x = right_col_x - _col_hw(right_col_x)
                    for ls in link_styles:
                        coords = ls.get("coordinates", [])
                        for k in range(len(coords)):
                            if coords[k][0] >= threshold_x:
                                coords[k] = [coords[k][0] + shift, coords[k][1]]

                    shifted = True
                    break  # restart — do NOT assign X values yet

                # --- 4. Gap is large enough — assign equally-spaced X to GROUPS ---
                # Each group (merged segments) gets ONE rank, ONE new X.
                # Sort by: path type (Z first), then X coordinate
                sorted_x_values = sorted(
                    x_groups.keys(),
                    key=lambda x: (0 if x_types[x] == "Z" else 1, x)
                )
                for rank, old_x in enumerate(sorted_x_values):
                    new_x = base_x + VLINE_INCREMENT * rank
                    for seg in x_groups[old_x]:
                        coords = link_styles[seg["link_idx"]]["coordinates"]
                        coords[seg["seg_idx"]] = [new_x, coords[seg["seg_idx"]][1]]
                        coords[seg["seg_idx"] + 1] = [new_x, coords[seg["seg_idx"] + 1][1]]

            if not shifted:
                break

        # =====================================================================
        # Step 1B: Compress wasted vertical space (column gaps) — SHRINK pass
        # After Step 1, every column gap satisfies distance >= LAST_VLINE_GAP and
        # waypoints are packed into [base_x, rightmost_x]. Some gaps may have
        # MORE space than needed (distance > LAST_VLINE_GAP), wasting horizontal
        # space. Here we shrink such gaps so distance == LAST_VLINE_GAP.
        #
        # Why shrinking is safe here (no U-shape risk):
        # - Step 1 has packed waypoints to [base_x, rightmost_x]
        # - rightmost_x = base_x + (n_groups - 1) * VLINE_INCREMENT
        # - rightmost_x < right_col_x_old - ICON_HALF (because distance >= LAST_VLINE_GAP > 0)
        # - threshold_x = right_col_x_old - ICON_HALF
        # - waypoint x in [base_x, rightmost_x] < threshold_x, so they are NOT shifted
        # - new right_col_x = ideal = rightmost_x + LAST_VLINE_GAP + ICON_HALF
        # - waypoint x <= rightmost_x < new right_col_x, so still inside the new gap
        # =====================================================================
        for _iteration in range(MAX_ITER):
            unit_x_coords = sorted(
                set(int(u.get("position_x", 0)) for u in layout
                    if u.get("position_x") is not None)
            )
            if len(unit_x_coords) < 2:
                break

            shifted = False

            for col_idx in range(len(unit_x_coords) - 1):
                left_col_x = unit_x_coords[col_idx]
                right_col_x = unit_x_coords[col_idx + 1]

                # --- 1. Extract vertical segments in this gap ---
                v_segments = []
                for link_idx, link in enumerate(link_styles):
                    coords = link.get("coordinates", [])
                    for seg_idx in range(len(coords) - 1):
                        if coords[seg_idx][0] == coords[seg_idx + 1][0]:
                            x = coords[seg_idx][0]
                            if left_col_x + _col_hw(left_col_x) < x < right_col_x - _col_hw(right_col_x):
                                v_segments.append({
                                    "link_idx": link_idx,
                                    "seg_idx": seg_idx,
                                    "x": x,
                                })

                # --- 2. Group by X (merged segments share one slot) ---
                # Note: empty gaps (no v_segments) are NOT skipped here — they
                # should still be compressed down to LAYOUT_MIN_COL_SPACING so
                # that minimum column spacing is enforced uniformly across the
                # whole layout (otherwise a column gap with no waypoints keeps
                # whatever spacing pre-processing left behind, which can be
                # much larger than LAYOUT_MIN_COL_SPACING).
                x_groups = defaultdict(list)
                for seg in v_segments:
                    x_groups[seg["x"]].append(seg)

                n_groups = len(x_groups)

                # --- 3. Compute compact ideal right column position ---
                # The ideal column position must satisfy BOTH constraints:
                #   (a) Line spacing rule (when there are segments):
                #       ideal >= rightmost_x + LAST_VLINE_GAP + (right col half-width)
                #   (b) Minimum column spacing (always):
                #       ideal >= left_col_x + LAYOUT_MIN_COL_SPACING
                # When n_groups == 0 (empty gap) only (b) applies.
                if n_groups > 0:
                    base_x = _snap_to_grid(int(left_col_x + _col_hw(left_col_x) + FIRST_VLINE_GAP))
                    rightmost_x = base_x + VLINE_INCREMENT * (n_groups - 1)
                    line_ideal_x = rightmost_x + LAST_VLINE_GAP + _col_hw(right_col_x)
                else:
                    line_ideal_x = 0  # no segments — only minimum spacing matters

                ideal_right_col_x = max(
                    line_ideal_x,
                    left_col_x + LAYOUT_MIN_COL_SPACING,
                )

                # --- 4. If wasted space exists, shrink right column ---
                if right_col_x > ideal_right_col_x:
                    shift = ideal_right_col_x - right_col_x  # negative (shrink)

                    for u in layout:
                        ux = int(u.get("position_x", 0))
                        if ux >= right_col_x:
                            u["position_x"] = ux + shift

                    threshold_x = right_col_x - _col_hw(right_col_x)
                    for ls in link_styles:
                        coords = ls.get("coordinates", [])
                        for k in range(len(coords)):
                            if coords[k][0] >= threshold_x:
                                coords[k] = [coords[k][0] + shift, coords[k][1]]

                    shifted = True
                    break  # restart

            if not shifted:
                break

        # =====================================================================
        # Step 2: Process horizontal segments (row gaps)
        # MIRROR of _post_process_horizontal_segments. Only difference:
        # segments at the same Y are merged into one group (n_groups vs n_segs).
        # =====================================================================
        for _iteration in range(MAX_ITER):
            unit_y_coords = sorted(
                set(int(u.get("position_y", 0)) for u in layout
                    if u.get("position_y") is not None)
            )
            if len(unit_y_coords) < 2:
                break

            shifted = False

            for row_idx in range(len(unit_y_coords) - 1):
                upper_row_y = unit_y_coords[row_idx]
                lower_row_y = unit_y_coords[row_idx + 1]

                # --- 1. Extract horizontal segments in this gap ---
                h_segments = []
                for link_idx, link in enumerate(link_styles):
                    coords = link.get("coordinates", [])
                    for seg_idx in range(len(coords) - 1):
                        if coords[seg_idx][1] == coords[seg_idx + 1][1]:
                            y = coords[seg_idx][1]
                            if upper_row_y + _bottom_off(upper_row_y) < y < lower_row_y - _top_off(lower_row_y):
                                h_segments.append({
                                    "link_idx": link_idx,
                                    "seg_idx": seg_idx,
                                    "y": y,
                                })

                if not h_segments:
                    continue

                # --- 2. Group by Y coordinate (CORE DIFFERENCE from pre-processing) ---
                y_groups = defaultdict(list)
                for seg in h_segments:
                    y_groups[seg["y"]].append(seg)

                n_groups = len(y_groups)
                if n_groups == 0:
                    continue

                # --- 3. Check if gap is large enough BEFORE assigning Y ---
                # NOTE: Do NOT snap to grid=20 (see _post_process_horizontal_segments
                # for rationale). Bottom_edge + FIRST_HLINE_GAP is already aligned
                # to the 16-grid that unit cy follows. Uses each row's ACTUAL edge
                # offsets (variable icon sizes) instead of 40x40's 10/30.
                base_y = int(upper_row_y + _bottom_off(upper_row_y) + FIRST_HLINE_GAP)
                bottommost_y = base_y + HLINE_INCREMENT * (n_groups - 1)
                lower_top_edge = lower_row_y - _top_off(lower_row_y)
                distance = lower_top_edge - bottommost_y

                if distance < LAST_HLINE_GAP:
                    # Gap too small — shift lower units down FIRST
                    # Round shift up to the nearest multiple of 16 to keep grid alignment
                    shift = ((LAST_HLINE_GAP - distance + 15) // 16) * 16

                    for u in layout:
                        uy = int(u.get("position_y", 0))
                        if uy >= lower_row_y:
                            u["position_y"] = uy + shift

                    threshold_y = lower_row_y - _top_off(lower_row_y)
                    for ls in link_styles:
                        coords = ls.get("coordinates", [])
                        for k in range(len(coords)):
                            if coords[k][1] >= threshold_y:
                                coords[k] = [coords[k][0], coords[k][1] + shift]

                    shifted = True
                    break  # restart — do NOT assign Y values yet

                # --- 4. Gap is large enough — assign equally-spaced Y to GROUPS ---
                sorted_y_values = sorted(y_groups.keys())
                for rank, old_y in enumerate(sorted_y_values):
                    new_y = base_y + HLINE_INCREMENT * rank
                    for seg in y_groups[old_y]:
                        link_idx = seg["link_idx"]
                        seg_idx = seg["seg_idx"]
                        coords = link_styles[link_idx]["coordinates"]

                        # Save old Y coordinate before updating
                        old_seg_y = coords[seg_idx][1]

                        # Update horizontal segment endpoints
                        coords[seg_idx] = [coords[seg_idx][0], new_y]
                        coords[seg_idx + 1] = [coords[seg_idx + 1][0], new_y]

                        # Synchronize adjacent start/end points to keep connections horizontal
                        # This prevents diagonal lines after _reassign_edge_points_after_merge
                        # has assigned different Y coordinates to start/end points
                        if seg_idx == 1 and len(coords) > 1:
                            # This is the second segment (start point → waypoint1)
                            # If start point has same Y as this segment, update it too
                            if coords[0][1] == old_seg_y:
                                coords[0] = [coords[0][0], new_y]

                        if seg_idx + 2 == len(coords) and len(coords) > 2:
                            # This is the second-to-last segment (waypoint_n → end point)
                            # If end point has same Y as this segment, update it too
                            if coords[-1][1] == old_seg_y:
                                coords[-1] = [coords[-1][0], new_y]

            if not shifted:
                break

        # =====================================================================
        # Step 2B: Compress wasted horizontal space (row gaps) — SHRINK pass
        # After Step 2, every row gap satisfies distance >= LAST_HLINE_GAP and
        # waypoints are packed into [base_y, bottommost_y]. Some gaps may have
        # MORE space than needed (distance > LAST_HLINE_GAP). Here we shrink
        # such gaps so distance == LAST_HLINE_GAP.
        #
        # Why shrinking is safe here (no U-shape risk): same reasoning as Step 1B.
        # Step 2 has packed waypoints; threshold_y > bottommost_y, so waypoints
        # are not affected by the shrink shift.
        # =====================================================================
        for _iteration in range(MAX_ITER):
            unit_y_coords = sorted(
                set(int(u.get("position_y", 0)) for u in layout
                    if u.get("position_y") is not None)
            )
            if len(unit_y_coords) < 2:
                break

            shifted = False

            for row_idx in range(len(unit_y_coords) - 1):
                upper_row_y = unit_y_coords[row_idx]
                lower_row_y = unit_y_coords[row_idx + 1]

                # --- 1. Extract horizontal segments in this gap ---
                h_segments = []
                for link_idx, link in enumerate(link_styles):
                    coords = link.get("coordinates", [])
                    for seg_idx in range(len(coords) - 1):
                        if coords[seg_idx][1] == coords[seg_idx + 1][1]:
                            y = coords[seg_idx][1]
                            if upper_row_y + _bottom_off(upper_row_y) < y < lower_row_y - _top_off(lower_row_y):
                                h_segments.append({
                                    "link_idx": link_idx,
                                    "seg_idx": seg_idx,
                                    "y": y,
                                })

                # --- 2. Group by Y (merged segments share one slot) ---
                # Note: empty gaps (no h_segments) are NOT skipped here — they
                # should still be compressed down to LAYOUT_MIN_ROW_SPACING so
                # that minimum row spacing is enforced uniformly across the
                # whole layout (otherwise a row gap with no waypoints keeps
                # whatever spacing pre-processing left behind, which can be
                # much larger than LAYOUT_MIN_ROW_SPACING).
                y_groups = defaultdict(list)
                for seg in h_segments:
                    y_groups[seg["y"]].append(seg)

                n_groups = len(y_groups)

                # --- 3. Compute compact ideal lower row position ---
                # The ideal row position must satisfy BOTH constraints:
                #   (a) Line spacing rule (when there are segments):
                #       ideal >= bottommost_y + LAST_HLINE_GAP + (lower row top offset)
                #   (b) Minimum row spacing (always):
                #       ideal >= upper_row_y + LAYOUT_MIN_ROW_SPACING
                # When n_groups == 0 (empty gap) only (b) applies.
                if n_groups > 0:
                    # NOTE: Do NOT snap to grid=20. Bottom_edge + FIRST_HLINE_GAP
                    # is already on the 16-grid; snapping would break gap uniformity.
                    base_y = int(upper_row_y + _bottom_off(upper_row_y) + FIRST_HLINE_GAP)
                    bottommost_y = base_y + HLINE_INCREMENT * (n_groups - 1)
                    line_ideal_y = bottommost_y + LAST_HLINE_GAP + _top_off(lower_row_y)
                else:
                    line_ideal_y = 0  # no segments — only minimum spacing matters

                ideal_lower_row_y = max(
                    line_ideal_y,
                    upper_row_y + LAYOUT_MIN_ROW_SPACING,
                )

                # --- 4. If wasted space exists, shrink lower row ---
                if lower_row_y > ideal_lower_row_y:
                    shift = ideal_lower_row_y - lower_row_y  # negative (shrink)
                    # Round shift toward zero in multiples of 16 to stay on grid
                    # Since shift is negative: -((|shift|) // 16) * 16
                    shift = -(((-shift) // 16) * 16)

                    if shift == 0:
                        continue

                    for u in layout:
                        uy = int(u.get("position_y", 0))
                        if uy >= lower_row_y:
                            u["position_y"] = uy + shift

                    threshold_y = lower_row_y - _top_off(lower_row_y)
                    for ls in link_styles:
                        coords = ls.get("coordinates", [])
                        for k in range(len(coords)):
                            if coords[k][1] >= threshold_y:
                                coords[k] = [coords[k][0], coords[k][1] + shift]

                    shifted = True
                    break  # restart

            if not shifted:
                break

        layout_data["layout"] = layout
        layout_data["link_styles"] = link_styles
        return layout_data
