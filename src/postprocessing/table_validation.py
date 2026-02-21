"""
Table validation and cell extraction for OCR output.

Handles structural validation of detected tables, filtering of false positives,
and extraction of cell-level content from Table Transformer structure data.
"""

import logging
from typing import Dict, List, Tuple

from src.utils.bbox import bbox_overlap_ratio_of_smaller, split_line_bbox_to_words, estimate_page_dimensions
from src.config import CONFIG

logger = logging.getLogger(__name__)

# ============================================================================
# Structural Validation Filter for Tables
# ============================================================================

# Configuration thresholds (tunable)
MIN_TEXT_DENSITY = 0.03  # 3% minimum text coverage
MAX_TEXT_DENSITY = 0.80  # 80% maximum (too dense = overlapping)
DEFAULT_COLUMN_CLUSTER_TOLERANCE = 40  # Pixels for column clustering at reference width
DEFAULT_ROW_CLUSTER_TOLERANCE = 15     # Pixels for row clustering at reference width
REFERENCE_PAGE_WIDTH = 1000            # Reference page width for tolerance scaling


def _scaled_tolerances(elements: List[Dict]) -> Tuple[float, float]:
    """Compute column/row clustering tolerances scaled to actual page width.

    Uses the shared ``estimate_page_dimensions`` utility to determine page
    width, then scales the default tolerances proportionally.

    Returns:
        (col_tolerance, row_tolerance) scaled to actual page width
    """
    page_width, _ = estimate_page_dimensions(elements)
    if page_width <= 0:
        return DEFAULT_COLUMN_CLUSTER_TOLERANCE, DEFAULT_ROW_CLUSTER_TOLERANCE

    scale = page_width / REFERENCE_PAGE_WIDTH
    return DEFAULT_COLUMN_CLUSTER_TOLERANCE * scale, DEFAULT_ROW_CLUSTER_TOLERANCE * scale


def filter_empty_regions(elements: List[Dict]) -> List[Dict]:
    """
    Filter out table and layout_region elements that contain no text.

    A region is kept only if at least one text element's bbox overlaps
    with it by >= 30%. This removes hallucinated regions in empty/noisy areas.

    Args:
        elements: List of element dictionaries

    Returns:
        Filtered list with empty regions removed
    """
    min_overlap_ratio = CONFIG.min_overlap_ratio

    # Separate text elements from regions
    text_elements = [e for e in elements if e.get("type") == "text"]
    text_bboxes = [e.get("bbox", []) for e in text_elements]

    filtered = []
    removed_count = 0

    for element in elements:
        elem_type = element.get("type", "")

        # Keep text elements and other types (logo, etc.)
        if elem_type not in ("table", "layout_region"):
            filtered.append(element)
            continue

        # Check if this region overlaps with any text
        region_bbox = element.get("bbox", [])
        if not region_bbox:
            removed_count += 1
            continue

        has_text_overlap = False
        for text_bbox in text_bboxes:
            if text_bbox and bbox_overlap_ratio_of_smaller(region_bbox, text_bbox) >= min_overlap_ratio:
                has_text_overlap = True
                break

        if has_text_overlap:
            filtered.append(element)
        else:
            removed_count += 1

    return filtered


def calculate_text_density(table_bbox: List[float], text_bboxes: List[List[float]]) -> float:
    """
    Calculate ratio of text area to table area.
    """
    if len(table_bbox) != 4:
        return 0.0

    table_area = (table_bbox[2] - table_bbox[0]) * (table_bbox[3] - table_bbox[1])
    if table_area <= 0:
        return 0.0

    total_text_area = 0.0
    for bbox in text_bboxes:
        if len(bbox) >= 4:
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            if width > 0 and height > 0:
                total_text_area += width * height

    return min(total_text_area / table_area, 1.0)


def cluster_positions(positions: List[float], tolerance: float) -> List[List[float]]:
    """
    Cluster similar positions together.
    """
    if not positions:
        return []

    sorted_pos = sorted(positions)
    clusters = [[sorted_pos[0]]]

    for pos in sorted_pos[1:]:
        if abs(pos - clusters[-1][-1]) <= tolerance:
            clusters[-1].append(pos)
        else:
            clusters.append([pos])

    return clusters


def detect_column_alignment(table_bbox: List[float], text_bboxes: List[List[float]],
                            col_tolerance: float = DEFAULT_COLUMN_CLUSTER_TOLERANCE) -> Tuple[int, List[float]]:
    """
    Detect column structure by clustering text X-positions.
    """
    if not text_bboxes:
        return 0, []

    x_positions = [bbox[0] for bbox in text_bboxes if len(bbox) >= 4]

    if not x_positions:
        return 0, []

    clusters = cluster_positions(x_positions, col_tolerance)
    column_centers = [sum(c) / len(c) for c in clusters]

    return len(clusters), column_centers


def detect_row_alignment(table_bbox: List[float], text_bboxes: List[List[float]],
                         row_tolerance: float = DEFAULT_ROW_CLUSTER_TOLERANCE) -> Tuple[int, List[float]]:
    """
    Detect row structure by clustering text Y-positions.
    """
    if not text_bboxes:
        return 0, []

    y_positions = [bbox[1] for bbox in text_bboxes if len(bbox) >= 4]

    if not y_positions:
        return 0, []

    clusters = cluster_positions(y_positions, row_tolerance)
    row_centers = [sum(c) / len(c) for c in clusters]

    return len(clusters), row_centers


def check_grid_coverage(text_bboxes: List[List[float]],
                        column_centers: List[float],
                        row_centers: List[float],
                        col_tolerance: float = DEFAULT_COLUMN_CLUSTER_TOLERANCE,
                        row_tolerance: float = DEFAULT_ROW_CLUSTER_TOLERANCE) -> float:
    """
    Check how well text elements cover the grid (columns x rows).
    """
    if len(column_centers) < 2 or len(row_centers) < 2:
        return 0.0

    total_cells = len(column_centers) * len(row_centers)

    occupied_cells = set()
    for bbox in text_bboxes:
        if len(bbox) < 4:
            continue

        text_x = (bbox[0] + bbox[2]) / 2
        text_y = (bbox[1] + bbox[3]) / 2

        col_idx = -1
        min_col_dist = float('inf')
        for i, col_x in enumerate(column_centers):
            dist = abs(text_x - col_x)
            if dist < min_col_dist and dist < col_tolerance * 2:
                min_col_dist = dist
                col_idx = i

        row_idx = -1
        min_row_dist = float('inf')
        for i, row_y in enumerate(row_centers):
            dist = abs(text_y - row_y)
            if dist < min_row_dist and dist < row_tolerance * 2:
                min_row_dist = dist
                row_idx = i

        if col_idx >= 0 and row_idx >= 0:
            occupied_cells.add((col_idx, row_idx))

    return len(occupied_cells) / total_cells if total_cells > 0 else 0.0


def calculate_structure_score(
    density: float,
    num_cols: int,
    num_rows: int,
    grid_coverage: float,
    confidence: float
) -> Tuple[float, List[str]]:
    """
    Calculate structural validity score for a table.
    """
    score = 0.0
    signals = []

    # Signal 1: Multi-column structure (40 points max)
    if num_cols >= 3:
        score += 40
        signals.append(f"columns:{num_cols}")
    elif num_cols == 2:
        score += 25
        signals.append(f"columns:{num_cols}")
    else:
        signals.append("single_column")

    # Signal 2: Row structure (25 points max)
    if num_rows >= 3:
        score += 25
        signals.append(f"rows:{num_rows}")
    elif num_rows == 2:
        score += 15
        signals.append(f"rows:{num_rows}")
    else:
        signals.append("few_rows")

    # Signal 3: Grid coverage (20 points max)
    if grid_coverage >= 0.3:
        score += 20
        signals.append(f"grid_coverage:{grid_coverage:.0%}")
    elif grid_coverage >= 0.15:
        score += 10
        signals.append(f"grid_coverage:{grid_coverage:.0%}")
    else:
        signals.append(f"sparse_grid:{grid_coverage:.0%}")

    # Signal 4: Text density in valid range (10 points)
    if MIN_TEXT_DENSITY <= density <= MAX_TEXT_DENSITY:
        score += 10
        signals.append(f"density_ok:{density:.1%}")
    else:
        signals.append(f"density_bad:{density:.1%}")

    # Signal 5: Model confidence bonus (5 points)
    if confidence >= 0.95:
        score += 5
        signals.append("high_confidence")

    return score, signals


def validate_table_structure(
    table_element: Dict,
    text_elements: List[Dict],
    col_tolerance: float = DEFAULT_COLUMN_CLUSTER_TOLERANCE,
    row_tolerance: float = DEFAULT_ROW_CLUSTER_TOLERANCE,
) -> Tuple[bool, float, List[str]]:
    """
    Validate if a detected table has true tabular structure.
    """
    table_bbox = table_element.get("bbox", [])
    confidence = table_element.get("confidence", 0.5)

    if len(table_bbox) != 4:
        return False, 0.0, ["invalid_bbox"]

    # Get text elements that overlap with this table
    overlapping_texts = []
    for text_elem in text_elements:
        text_bbox = text_elem.get("bbox", [])
        if text_bbox and bbox_overlap_ratio_of_smaller(table_bbox, text_bbox) >= 0.3:
            overlapping_texts.append(text_bbox)

    if not overlapping_texts:
        return False, 0.0, ["no_text_overlap"]

    density = calculate_text_density(table_bbox, overlapping_texts)
    num_cols, col_centers = detect_column_alignment(table_bbox, overlapping_texts, col_tolerance)
    num_rows, row_centers = detect_row_alignment(table_bbox, overlapping_texts, row_tolerance)
    grid_coverage = check_grid_coverage(overlapping_texts, col_centers, row_centers,
                                        col_tolerance, row_tolerance)

    score, signals = calculate_structure_score(
        density, num_cols, num_rows, grid_coverage, confidence
    )

    is_valid = score >= CONFIG.min_structure_score

    return is_valid, score, signals


def filter_invalid_tables(elements: List[Dict]) -> List[Dict]:
    """
    Filter out tables that don't have valid tabular structure.
    """
    text_elements = [e for e in elements if e.get("type") == "text"]
    col_tol, row_tol = _scaled_tolerances(elements)

    filtered = []

    for element in elements:
        if element.get("type") != "table":
            filtered.append(element)
            continue

        is_valid, score, signals = validate_table_structure(
            element, text_elements, col_tol, row_tol
        )

        if is_valid:
            element["structure_score"] = round(score, 1)
            element["structure_signals"] = signals
            filtered.append(element)

    return filtered


# ============================================================================
# Table Cell Extraction (from Table Transformer structure data)
# ============================================================================

def build_table_cells(
    table_element: Dict,
    text_elements: List[Dict],
) -> List[Dict]:
    """Build cell grid from table structure and assign text content to cells.

    Uses the row/column bboxes from Table Transformer's structure recognition
    to create a grid of cells, then assigns OCR text to each cell based on
    word-level center-point containment.

    Args:
        table_element: Table element dict with ``structure`` key containing
            ``rows``, ``columns``, ``cells``, and ``headers`` lists.
        text_elements: All text elements on the page.

    Returns:
        List of cell dicts with ``row``, ``col``, ``bbox``, ``content``,
        and ``is_header`` keys.  Returns empty list if structure has
        0 rows or 0 columns.
    """
    structure = table_element.get("structure", {})
    rows = structure.get("rows", [])
    columns = structure.get("columns", [])
    headers = structure.get("headers", [])
    table_bbox = table_element.get("bbox", [])

    if not rows or not columns or len(table_bbox) != 4:
        return []

    # Sort rows top-to-bottom, columns left-to-right (should already be sorted)
    rows = sorted(rows, key=lambda r: r["bbox"][1])
    columns = sorted(columns, key=lambda c: c["bbox"][0])

    # Build cell grid: each cell = intersection of row y-range and column x-range
    # Gap-fill: extend adjacent rows/columns to the midpoint of the gap between
    # them so small structural gaps don't lose text.  For first/last row/column,
    # extend outward by at most half the median row height or column width to
    # avoid absorbing header text or content outside the table.
    row_heights = [r["bbox"][3] - r["bbox"][1] for r in rows]
    col_widths = [c["bbox"][2] - c["bbox"][0] for c in columns]
    median_row_h = sorted(row_heights)[len(row_heights) // 2] if row_heights else 0
    median_col_w = sorted(col_widths)[len(col_widths) // 2] if col_widths else 0

    # Pre-compute adjusted row y-ranges
    row_ranges = []
    for i, row in enumerate(rows):
        ry1, ry2 = row["bbox"][1], row["bbox"][3]
        # Extend top edge
        if i == 0:
            ry1 = max(ry1 - median_row_h / 2, table_bbox[1])
        else:
            prev_bottom = rows[i - 1]["bbox"][3]
            ry1 = (prev_bottom + ry1) / 2
        # Extend bottom edge
        if i == len(rows) - 1:
            ry2 = min(ry2 + median_row_h / 2, table_bbox[3])
        else:
            next_top = rows[i + 1]["bbox"][1]
            ry2 = (ry2 + next_top) / 2
        row_ranges.append((ry1, ry2))

    # Pre-compute adjusted column x-ranges
    col_ranges = []
    for j, col in enumerate(columns):
        cx1, cx2 = col["bbox"][0], col["bbox"][2]
        # Extend left edge
        if j == 0:
            cx1 = max(cx1 - median_col_w / 2, table_bbox[0])
        else:
            prev_right = columns[j - 1]["bbox"][2]
            cx1 = (prev_right + cx1) / 2
        # Extend right edge
        if j == len(columns) - 1:
            cx2 = min(cx2 + median_col_w / 2, table_bbox[2])
        else:
            next_left = columns[j + 1]["bbox"][0]
            cx2 = (cx2 + next_left) / 2
        col_ranges.append((cx1, cx2))

    cells: List[Dict] = []
    for row_idx, row in enumerate(rows):
        ry1, ry2 = row_ranges[row_idx]
        for col_idx, col in enumerate(columns):
            cx1, cx2 = col_ranges[col_idx]
            cell_bbox = [cx1, ry1, cx2, ry2]

            # Check if this cell overlaps any header region
            is_header = False
            for hdr in headers:
                hb = hdr["bbox"]
                # Simple overlap: header bbox intersects this cell
                if (hb[0] < cx2 and hb[2] > cx1 and
                        hb[1] < ry2 and hb[3] > ry1):
                    is_header = True
                    break

            cells.append({
                "row": row_idx,
                "col": col_idx,
                "bbox": [round(v, 2) for v in cell_bbox],
                "content": "",
                "is_header": is_header,
            })

    # Collect text elements overlapping the table (>=50%)
    overlapping_texts = []
    for te in text_elements:
        te_bbox = te.get("bbox", [])
        if te_bbox and bbox_overlap_ratio_of_smaller(table_bbox, te_bbox) >= 0.5:
            overlapping_texts.append(te)

    # Assign words to cells via center-point containment
    for te in overlapping_texts:
        content = te.get("content", "")
        te_bbox = te.get("bbox", [])
        if not content or not te_bbox:
            continue

        words = content.split()
        if not words:
            continue

        word_bboxes = split_line_bbox_to_words(te_bbox, words)

        for word, wb in zip(words, word_bboxes):
            # Word center point
            wcx = (wb[0] + wb[2]) / 2
            wcy = (wb[1] + wb[3]) / 2

            # Find which cell contains this center point
            for cell in cells:
                cb = cell["bbox"]
                if cb[0] <= wcx <= cb[2] and cb[1] <= wcy <= cb[3]:
                    if cell["content"]:
                        cell["content"] += " " + word
                    else:
                        cell["content"] = word
                    break

    return cells
