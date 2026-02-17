"""
Table validation and heuristic promotion for OCR output.

Handles structural validation of detected tables, filtering of false positives,
and promotion of layout region clusters to table elements.
"""

import logging
from typing import Dict, List, Tuple

from src.utils.bbox import bbox_overlap_ratio_of_smaller

logger = logging.getLogger(__name__)

# ============================================================================
# Structural Validation Filter for Tables
# ============================================================================

# Configuration thresholds (tunable)
MIN_COLUMNS = 2          # Minimum columns to be considered a table
MIN_ROWS = 2             # Minimum rows to be considered a table
MIN_TEXT_DENSITY = 0.03  # 3% minimum text coverage
MAX_TEXT_DENSITY = 0.80  # 80% maximum (too dense = overlapping)
MIN_STRUCTURE_SCORE = 50 # Minimum score to keep as table
DEFAULT_COLUMN_CLUSTER_TOLERANCE = 40  # Pixels for column clustering at reference width
DEFAULT_ROW_CLUSTER_TOLERANCE = 15     # Pixels for row clustering at reference width
REFERENCE_PAGE_WIDTH = 1000            # Reference page width for tolerance scaling


def _scaled_tolerances(elements: List[Dict]) -> Tuple[float, float]:
    """Compute column/row clustering tolerances scaled to actual page width.

    Uses the maximum x2 coordinate across all element bboxes as a proxy for
    page width, then scales the default tolerances proportionally.

    Returns:
        (col_tolerance, row_tolerance) scaled to actual page width
    """
    page_width = 0
    for element in elements:
        bbox = element.get("bbox", [])
        if len(bbox) >= 4:
            page_width = max(page_width, bbox[2])
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
    MIN_OVERLAP_RATIO = 0.3  # 30% overlap required

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
            if text_bbox and bbox_overlap_ratio_of_smaller(region_bbox, text_bbox) >= MIN_OVERLAP_RATIO:
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

    is_valid = score >= MIN_STRUCTURE_SCORE

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
# Heuristic Table Promotion (for missed borderless tables)
# ============================================================================

# Configuration for table promotion
VERTICAL_CLUSTER_GAP = 20
MIN_CLUSTER_REGIONS = 15
MIN_PROMOTION_COLUMNS = 4
MIN_PROMOTION_ROWS = 5
MIN_PROMOTION_SCORE = 90
MIN_GRID_COVERAGE = 0.55
TABLE_OVERLAP_THRESHOLD = 0.40

# Boundary refinement
MIN_ANCHOR_COLUMNS = 4
ANCHOR_ROW_HEIGHT = 18
HIGH_DENSITY_ITEMS = 6
MIN_CONSECUTIVE_COLS = 4


def cluster_layout_regions_vertically(elements: List[Dict]) -> List[List[Dict]]:
    """
    Cluster layout_region elements by vertical proximity.
    """
    layout_regions = [e for e in elements if e.get("type") == "layout_region"]

    if len(layout_regions) < MIN_CLUSTER_REGIONS:
        return []

    regions_with_y = []
    for region in layout_regions:
        bbox = region.get("bbox", [])
        if len(bbox) >= 4:
            y_top = bbox[1]
            regions_with_y.append((y_top, region))

    if not regions_with_y:
        return []

    regions_with_y.sort(key=lambda x: x[0])

    clusters = []
    current_cluster = [regions_with_y[0][1]]
    current_y_bottom = regions_with_y[0][1].get("bbox", [0, 0, 0, 0])[3]

    for y_top, region in regions_with_y[1:]:
        bbox = region.get("bbox", [0, 0, 0, 0])

        if y_top - current_y_bottom <= VERTICAL_CLUSTER_GAP:
            current_cluster.append(region)
            current_y_bottom = max(current_y_bottom, bbox[3])
        else:
            if len(current_cluster) >= MIN_CLUSTER_REGIONS:
                clusters.append(current_cluster)
            current_cluster = [region]
            current_y_bottom = bbox[3]

    if len(current_cluster) >= MIN_CLUSTER_REGIONS:
        clusters.append(current_cluster)

    return clusters


def calculate_cluster_bbox(cluster: List[Dict]) -> List[float]:
    """
    Calculate combined bounding box for a cluster of regions.
    """
    if not cluster:
        return []

    min_x = float('inf')
    min_y = float('inf')
    max_x = float('-inf')
    max_y = float('-inf')

    for region in cluster:
        bbox = region.get("bbox", [])
        if len(bbox) >= 4:
            min_x = min(min_x, bbox[0])
            min_y = min(min_y, bbox[1])
            max_x = max(max_x, bbox[2])
            max_y = max(max_y, bbox[3])

    if min_x == float('inf'):
        return []

    return [min_x, min_y, max_x, max_y]


def validate_cluster_as_table(
    cluster: List[Dict],
    text_elements: List[Dict],
    col_tolerance: float = DEFAULT_COLUMN_CLUSTER_TOLERANCE,
    row_tolerance: float = DEFAULT_ROW_CLUSTER_TOLERANCE,
) -> Tuple[bool, float, List[str]]:
    """
    Validate if a cluster of layout regions has valid table structure.
    """
    cluster_bbox = calculate_cluster_bbox(cluster)
    if not cluster_bbox:
        return False, 0.0, ["no_bbox"]

    overlapping_texts = []
    for text_elem in text_elements:
        text_bbox = text_elem.get("bbox", [])
        if text_bbox and bbox_overlap_ratio_of_smaller(cluster_bbox, text_bbox) >= 0.3:
            overlapping_texts.append(text_bbox)

    if len(overlapping_texts) < 6:
        return False, 0.0, ["insufficient_text"]

    density = calculate_text_density(cluster_bbox, overlapping_texts)
    num_cols, col_centers = detect_column_alignment(cluster_bbox, overlapping_texts, col_tolerance)
    num_rows, row_centers = detect_row_alignment(cluster_bbox, overlapping_texts, row_tolerance)
    grid_coverage = check_grid_coverage(overlapping_texts, col_centers, row_centers,
                                        col_tolerance, row_tolerance)

    score, signals = calculate_structure_score(
        density, num_cols, num_rows, grid_coverage, 0.80
    )

    is_valid = (
        score >= MIN_PROMOTION_SCORE and
        num_cols >= MIN_PROMOTION_COLUMNS and
        num_rows >= MIN_PROMOTION_ROWS and
        grid_coverage >= MIN_GRID_COVERAGE
    )

    return is_valid, score, signals


def overlaps_existing_table(cluster_bbox: List[float], tables: List[Dict]) -> bool:
    """
    Check if cluster overlaps significantly with any existing table.
    """
    if not cluster_bbox or len(cluster_bbox) != 4:
        return False

    for table in tables:
        table_bbox = table.get("bbox", [])
        if len(table_bbox) != 4:
            continue

        overlap = bbox_overlap_ratio_of_smaller(cluster_bbox, table_bbox)
        if overlap >= TABLE_OVERLAP_THRESHOLD:
            return True

    return False


def _analyze_row_structure(row_texts: List[List[float]], bbox_width: float,
                           col_tolerance: float = DEFAULT_COLUMN_CLUSTER_TOLERANCE) -> dict:
    """
    Analyze a row's structural properties.
    """
    result = {
        "num_items": len(row_texts),
        "num_cols": 0,
        "span_ratio": 0.0,
        "is_valid": False
    }

    if not row_texts:
        return result

    x_positions = [bbox[0] for bbox in row_texts]
    column_clusters = cluster_positions(x_positions, col_tolerance)
    result["num_cols"] = len(column_clusters)

    if len(column_clusters) >= 2:
        col_centers = sorted([sum(c) / len(c) for c in column_clusters])
        col_span = col_centers[-1] - col_centers[0]
        result["span_ratio"] = col_span / bbox_width if bbox_width > 0 else 0

    result["is_valid"] = (
        result["num_items"] >= MIN_ANCHOR_COLUMNS and
        result["num_cols"] >= MIN_ANCHOR_COLUMNS and
        result["span_ratio"] >= 0.60
    )

    return result


def refine_table_top_boundary(
    cluster_bbox: List[float],
    text_elements: List[Dict],
    col_tolerance: float = DEFAULT_COLUMN_CLUSTER_TOLERANCE,
) -> List[float]:
    """
    Refine the top boundary of a promoted table by trimming header rows
    until a row with proper table-like multi-column alignment is found.
    """
    if not cluster_bbox or len(cluster_bbox) != 4:
        return cluster_bbox

    x1, y1, x2, y2 = cluster_bbox
    bbox_width = x2 - x1

    if bbox_width <= 0:
        return cluster_bbox

    overlapping_texts = []
    for text_elem in text_elements:
        text_bbox = text_elem.get("bbox", [])
        if text_bbox and bbox_overlap_ratio_of_smaller(cluster_bbox, text_bbox) >= 0.3:
            overlapping_texts.append(text_bbox)

    if len(overlapping_texts) < 6:
        return cluster_bbox

    overlapping_texts.sort(key=lambda b: b[1])

    # Cluster into rows
    rows = []
    current_row = [overlapping_texts[0]]
    current_row_y = overlapping_texts[0][1]

    for text_bbox in overlapping_texts[1:]:
        text_y = text_bbox[1]
        if abs(text_y - current_row_y) <= ANCHOR_ROW_HEIGHT:
            current_row.append(text_bbox)
        else:
            rows.append(current_row)
            current_row = [text_bbox]
            current_row_y = text_y

    if current_row:
        rows.append(current_row)

    row_analyses = [_analyze_row_structure(row, bbox_width, col_tolerance) for row in rows]

    anchor_y = y1

    for i, (row_texts, analysis) in enumerate(zip(rows, row_analyses)):
        if not row_texts:
            continue

        # RULE 1: High density rows are automatic anchors
        if analysis["num_items"] >= HIGH_DENSITY_ITEMS:
            anchor_y = min(bbox[1] for bbox in row_texts)
            break

        # RULE 2: Consecutive structure
        if analysis["is_valid"]:
            if i + 1 < len(row_analyses):
                next_analysis = row_analyses[i + 1]
                if next_analysis["is_valid"] or next_analysis["num_items"] >= HIGH_DENSITY_ITEMS:
                    anchor_y = min(bbox[1] for bbox in row_texts)
                    break
            elif i == len(rows) - 1:
                anchor_y = min(bbox[1] for bbox in row_texts)
                break

    if anchor_y > y1 + ANCHOR_ROW_HEIGHT:
        return [x1, anchor_y, x2, y2]

    return cluster_bbox


def promote_layout_regions_to_tables(elements: List[Dict]) -> List[Dict]:
    """
    Promote clusters of layout_region elements to table elements.
    """
    existing_tables = [e for e in elements if e.get("type") == "table"]
    text_elements = [e for e in elements if e.get("type") == "text"]
    col_tol, row_tol = _scaled_tolerances(elements)

    clusters = cluster_layout_regions_vertically(elements)

    promoted_tables = []

    for cluster in clusters:
        cluster_bbox = calculate_cluster_bbox(cluster)

        if not cluster_bbox:
            continue

        if overlaps_existing_table(cluster_bbox, existing_tables):
            continue

        is_valid, score, signals = validate_cluster_as_table(
            cluster, text_elements, col_tol, row_tol
        )

        if is_valid:
            refined_bbox = refine_table_top_boundary(cluster_bbox, text_elements, col_tol)

            synthetic_table = {
                "type": "table",
                "bbox": refined_bbox,
                "confidence": 0.80,
                "source": "heuristic_promotion",
                "structure_score": round(score, 1),
                "structure_signals": signals
            }
            promoted_tables.append(synthetic_table)

    if promoted_tables:
        elements = elements + promoted_tables

    return elements
