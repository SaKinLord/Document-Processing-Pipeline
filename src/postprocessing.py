"""
Post-processing module for OCR output.

Implements multi-signal hallucination detection and output cleaning.
"""

import re
from typing import Dict, List, Any, Tuple, Optional


def normalize_punctuation_spacing(text: str) -> str:
    """
    Normalize spacing around punctuation in OCR output.
    
    TrOCR tends to add unnecessary spaces around punctuation marks which
    inflates WER (Word Error Rate) scores. This function fixes common patterns:
    - 'Govr. ,' -> 'Govr.,'
    - 'word . word' -> 'word. word'
    - '( text )' -> '(text)'
    - 'word ; word' -> 'word; word'
    
    Args:
        text: Raw OCR text output
        
    Returns:
        Text with normalized punctuation spacing
    """
    if not text:
        return text
    
    # Remove space before punctuation: 'word .' -> 'word.'
    text = re.sub(r'\s+([.,;:!?])', r'\1', text)
    
    # Remove space after opening brackets: '( text' -> '(text'
    text = re.sub(r'([\(\[\{])\s+', r'\1', text)
    
    # Remove space before closing brackets: 'text )' -> 'text)'
    text = re.sub(r'\s+([\)\]\}])', r'\1', text)
    
    # Fix double punctuation with space: '. ,' -> '.,'
    text = re.sub(r'([.,;:!?])\s+([.,;:!?])', r'\1\2', text)
    
    # Fix space around hyphens in words: 'self - aware' -> 'self-aware'
    text = re.sub(r'(\w)\s+-\s+(\w)', r'\1-\2', text)
    
    # Fix space around apostrophes: "don 't" -> "don't"
    text = re.sub(r"(\w)\s+'\s*(\w)", r"\1'\2", text)
    text = re.sub(r"(\w)\s+'", r"\1'", text)
    text = re.sub(r"'\s+(\w)", r"'\1", text)
    
    # Collapse multiple spaces into single space
    text = re.sub(r' {2,}', ' ', text)
    
    return text.strip()


def bbox_overlap(bbox1: List[float], bbox2: List[float]) -> float:
    """
    Calculate the overlap ratio between two bounding boxes.
    
    Args:
        bbox1: First bbox [x1, y1, x2, y2]
        bbox2: Second bbox [x1, y1, x2, y2]
        
    Returns:
        Overlap ratio (0.0 to 1.0) relative to the smaller bbox
    """
    if len(bbox1) != 4 or len(bbox2) != 4:
        return 0.0
    
    # Calculate intersection
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0  # No intersection
    
    intersection_area = (x2 - x1) * (y2 - y1)
    
    # Calculate areas
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    
    if area1 <= 0 or area2 <= 0:
        return 0.0
    
    # Return ratio relative to the smaller bbox (typically the text element)
    smaller_area = min(area1, area2)
    return intersection_area / smaller_area


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
            if text_bbox and bbox_overlap(region_bbox, text_bbox) >= MIN_OVERLAP_RATIO:
                has_text_overlap = True
                break
        
        if has_text_overlap:
            filtered.append(element)
        else:
            removed_count += 1
    
    return filtered


# ============================================================================
# Structural Validation Filter for Tables
# ============================================================================

# Configuration thresholds (tunable)
MIN_COLUMNS = 2          # Minimum columns to be considered a table
MIN_ROWS = 2             # Minimum rows to be considered a table
MIN_TEXT_DENSITY = 0.03  # 3% minimum text coverage
MAX_TEXT_DENSITY = 0.80  # 80% maximum (too dense = overlapping)
MIN_STRUCTURE_SCORE = 50 # Minimum score to keep as table
COLUMN_CLUSTER_TOLERANCE = 40  # Pixels for column clustering
ROW_CLUSTER_TOLERANCE = 15     # Pixels for row clustering


def calculate_text_density(table_bbox: List[float], text_bboxes: List[List[float]]) -> float:
    """
    Calculate ratio of text area to table area.
    
    Args:
        table_bbox: Table bounding box [x1, y1, x2, y2]
        text_bboxes: List of text bounding boxes inside the table
        
    Returns:
        Density ratio (0.0 to 1.0)
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
    
    Args:
        positions: List of coordinate values
        tolerance: Maximum distance to be in same cluster
        
    Returns:
        List of clusters (each cluster is list of positions)
    """
    if not positions:
        return []
    
    sorted_pos = sorted(positions)
    clusters = [[sorted_pos[0]]]
    
    for pos in sorted_pos[1:]:
        # Check if this position belongs to the last cluster
        if abs(pos - clusters[-1][-1]) <= tolerance:
            clusters[-1].append(pos)
        else:
            clusters.append([pos])
    
    return clusters


def detect_column_alignment(table_bbox: List[float], text_bboxes: List[List[float]]) -> Tuple[int, List[float]]:
    """
    Detect column structure by clustering text X-positions.
    
    Args:
        table_bbox: Table bounding box
        text_bboxes: Text bboxes inside the table
        
    Returns:
        Tuple of (num_columns, column_center_positions)
    """
    if not text_bboxes:
        return 0, []
    
    # Get left edge (x1) of each text element
    x_positions = [bbox[0] for bbox in text_bboxes if len(bbox) >= 4]
    
    if not x_positions:
        return 0, []
    
    clusters = cluster_positions(x_positions, COLUMN_CLUSTER_TOLERANCE)
    
    # Get center position of each cluster
    column_centers = [sum(c) / len(c) for c in clusters]
    
    return len(clusters), column_centers


def detect_row_alignment(table_bbox: List[float], text_bboxes: List[List[float]]) -> Tuple[int, List[float]]:
    """
    Detect row structure by clustering text Y-positions.
    
    Args:
        table_bbox: Table bounding box
        text_bboxes: Text bboxes inside the table
        
    Returns:
        Tuple of (num_rows, row_center_positions)
    """
    if not text_bboxes:
        return 0, []
    
    # Get top edge (y1) of each text element
    y_positions = [bbox[1] for bbox in text_bboxes if len(bbox) >= 4]
    
    if not y_positions:
        return 0, []
    
    clusters = cluster_positions(y_positions, ROW_CLUSTER_TOLERANCE)
    
    # Get center position of each cluster
    row_centers = [sum(c) / len(c) for c in clusters]
    
    return len(clusters), row_centers


def check_grid_coverage(text_bboxes: List[List[float]], 
                        column_centers: List[float], 
                        row_centers: List[float]) -> float:
    """
    Check how well text elements cover the grid (columns x rows).
    
    A real table should have text in multiple cells across the grid.
    
    Returns:
        Coverage ratio (0.0 to 1.0)
    """
    if len(column_centers) < 2 or len(row_centers) < 2:
        return 0.0
    
    # Create a grid of expected cells
    total_cells = len(column_centers) * len(row_centers)
    
    # Count occupied cells
    occupied_cells = set()
    for bbox in text_bboxes:
        if len(bbox) < 4:
            continue
        
        text_x = (bbox[0] + bbox[2]) / 2  # Center X
        text_y = (bbox[1] + bbox[3]) / 2  # Center Y
        
        # Find which column this text belongs to
        col_idx = -1
        min_col_dist = float('inf')
        for i, col_x in enumerate(column_centers):
            dist = abs(text_x - col_x)
            if dist < min_col_dist and dist < COLUMN_CLUSTER_TOLERANCE * 2:
                min_col_dist = dist
                col_idx = i
        
        # Find which row this text belongs to
        row_idx = -1
        min_row_dist = float('inf')
        for i, row_y in enumerate(row_centers):
            dist = abs(text_y - row_y)
            if dist < min_row_dist and dist < ROW_CLUSTER_TOLERANCE * 2:
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
    
    Args:
        density: Text density ratio
        num_cols: Number of detected columns
        num_rows: Number of detected rows
        grid_coverage: Grid coverage ratio
        confidence: Model confidence
        
    Returns:
        Tuple of (score 0-100, list of contributing signals)
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
    text_elements: List[Dict]
) -> Tuple[bool, float, List[str]]:
    """
    Validate if a detected table has true tabular structure.
    
    Args:
        table_element: Table element dictionary
        text_elements: All text elements on the page
        
    Returns:
        Tuple of (is_valid, score, signals)
    """
    table_bbox = table_element.get("bbox", [])
    confidence = table_element.get("confidence", 0.5)
    
    if len(table_bbox) != 4:
        return False, 0.0, ["invalid_bbox"]
    
    # Get text elements that overlap with this table
    overlapping_texts = []
    for text_elem in text_elements:
        text_bbox = text_elem.get("bbox", [])
        if text_bbox and bbox_overlap(table_bbox, text_bbox) >= 0.3:
            overlapping_texts.append(text_bbox)
    
    if not overlapping_texts:
        return False, 0.0, ["no_text_overlap"]
    
    # Calculate structural metrics
    density = calculate_text_density(table_bbox, overlapping_texts)
    num_cols, col_centers = detect_column_alignment(table_bbox, overlapping_texts)
    num_rows, row_centers = detect_row_alignment(table_bbox, overlapping_texts)
    grid_coverage = check_grid_coverage(overlapping_texts, col_centers, row_centers)
    
    # Calculate combined score
    score, signals = calculate_structure_score(
        density, num_cols, num_rows, grid_coverage, confidence
    )
    
    is_valid = score >= MIN_STRUCTURE_SCORE
    
    return is_valid, score, signals


def filter_invalid_tables(elements: List[Dict]) -> List[Dict]:
    """
    Filter out tables that don't have valid tabular structure.
    
    Tables with score < MIN_STRUCTURE_SCORE are removed.
    Tables with borderline scores are flagged.
    
    Args:
        elements: List of element dictionaries
        
    Returns:
        Filtered list with invalid tables removed
    """
    # Separate text elements for analysis
    text_elements = [e for e in elements if e.get("type") == "text"]
    
    filtered = []
    tables_removed = 0
    tables_kept = 0
    
    for element in elements:
        if element.get("type") != "table":
            filtered.append(element)
            continue
        
        # Validate table structure
        is_valid, score, signals = validate_table_structure(element, text_elements)
        
        if is_valid:
            # Add validation info to table
            element["structure_score"] = round(score, 1)
            element["structure_signals"] = signals
            filtered.append(element)
            tables_kept += 1
        else:
            # Remove invalid table
            tables_removed += 1
    
    return filtered


# ============================================================================
# Heuristic Table Promotion (for missed borderless tables)
# ============================================================================

# Configuration for table promotion (FINAL - very restrictive to avoid false positives)
VERTICAL_CLUSTER_GAP = 20        # Reduced gap for tighter clustering
MIN_CLUSTER_REGIONS = 15         # Require many regions (was 8) - only dense clusters
MIN_PROMOTION_COLUMNS = 4        # Require 4+ columns (was 3)
MIN_PROMOTION_ROWS = 5           # Require 5+ rows (was 3) - eliminates small headers
MIN_PROMOTION_SCORE = 90         # Very high score threshold (was 70)
MIN_GRID_COVERAGE = 0.55         # Require 55% grid coverage (was 35%)
TABLE_OVERLAP_THRESHOLD = 0.40   # Overlap ratio to consider as duplicate


def cluster_layout_regions_vertically(elements: List[Dict]) -> List[List[Dict]]:
    """
    Cluster layout_region elements by vertical proximity.
    
    Groups regions that are vertically adjacent (within VERTICAL_CLUSTER_GAP pixels).
    
    Args:
        elements: List of all elements
        
    Returns:
        List of clusters, where each cluster is a list of layout_region elements
    """
    # Extract layout regions only
    layout_regions = [e for e in elements if e.get("type") == "layout_region"]
    
    if len(layout_regions) < MIN_CLUSTER_REGIONS:
        return []
    
    # Sort by Y position (top of bbox)
    regions_with_y = []
    for region in layout_regions:
        bbox = region.get("bbox", [])
        if len(bbox) >= 4:
            y_top = bbox[1]
            regions_with_y.append((y_top, region))
    
    if not regions_with_y:
        return []
    
    regions_with_y.sort(key=lambda x: x[0])
    
    # Cluster by vertical proximity
    clusters = []
    current_cluster = [regions_with_y[0][1]]
    current_y_bottom = regions_with_y[0][1].get("bbox", [0, 0, 0, 0])[3]
    
    for y_top, region in regions_with_y[1:]:
        bbox = region.get("bbox", [0, 0, 0, 0])
        
        # Check if this region is close enough to current cluster
        if y_top - current_y_bottom <= VERTICAL_CLUSTER_GAP:
            current_cluster.append(region)
            current_y_bottom = max(current_y_bottom, bbox[3])
        else:
            # Start new cluster
            if len(current_cluster) >= MIN_CLUSTER_REGIONS:
                clusters.append(current_cluster)
            current_cluster = [region]
            current_y_bottom = bbox[3]
    
    # Don't forget the last cluster
    if len(current_cluster) >= MIN_CLUSTER_REGIONS:
        clusters.append(current_cluster)
    
    return clusters


def calculate_cluster_bbox(cluster: List[Dict]) -> List[float]:
    """
    Calculate combined bounding box for a cluster of regions.
    
    Args:
        cluster: List of layout_region elements
        
    Returns:
        Combined bbox [min_x, min_y, max_x, max_y]
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
    text_elements: List[Dict]
) -> Tuple[bool, float, List[str]]:
    """
    Validate if a cluster of layout regions has valid table structure.
    
    Uses text elements that overlap with the cluster to detect
    column and row alignment.
    
    Args:
        cluster: List of layout_region elements
        text_elements: All text elements on the page
        
    Returns:
        Tuple of (is_valid, score, signals)
    """
    cluster_bbox = calculate_cluster_bbox(cluster)
    if not cluster_bbox:
        return False, 0.0, ["no_bbox"]
    
    # Get text elements that overlap with this cluster
    overlapping_texts = []
    for text_elem in text_elements:
        text_bbox = text_elem.get("bbox", [])
        if text_bbox and bbox_overlap(cluster_bbox, text_bbox) >= 0.3:
            overlapping_texts.append(text_bbox)
    
    if len(overlapping_texts) < 6:
        return False, 0.0, ["insufficient_text"]
    
    # Calculate structural metrics
    density = calculate_text_density(cluster_bbox, overlapping_texts)
    num_cols, col_centers = detect_column_alignment(cluster_bbox, overlapping_texts)
    num_rows, row_centers = detect_row_alignment(cluster_bbox, overlapping_texts)
    grid_coverage = check_grid_coverage(overlapping_texts, col_centers, row_centers)
    
    # Calculate combined score (same logic as validate_table_structure)
    score, signals = calculate_structure_score(
        density, num_cols, num_rows, grid_coverage, 0.80  # Synthetic confidence
    )
    
    # Stricter requirements for promotion
    # Forms tend to have low grid coverage (label-value pattern)
    # Real tables have cells spanning the grid more evenly
    is_valid = (
        score >= MIN_PROMOTION_SCORE and 
        num_cols >= MIN_PROMOTION_COLUMNS and 
        num_rows >= MIN_PROMOTION_ROWS and
        grid_coverage >= MIN_GRID_COVERAGE  # Key: reject low-coverage forms
    )
    
    return is_valid, score, signals


def overlaps_existing_table(cluster_bbox: List[float], tables: List[Dict]) -> bool:
    """
    Check if cluster overlaps significantly with any existing table.
    
    Args:
        cluster_bbox: Bounding box of the cluster
        tables: List of existing table elements
        
    Returns:
        True if overlap > TABLE_OVERLAP_THRESHOLD with any table
    """
    if not cluster_bbox or len(cluster_bbox) != 4:
        return False
    
    for table in tables:
        table_bbox = table.get("bbox", [])
        if len(table_bbox) != 4:
            continue
        
        overlap = bbox_overlap(cluster_bbox, table_bbox)
        if overlap >= TABLE_OVERLAP_THRESHOLD:
            return True
    
    return False


# Minimum columns required to anchor the table top boundary
MIN_ANCHOR_COLUMNS = 4  # Real tables have 4+ columns
# Row height tolerance for row detection in refinement
ANCHOR_ROW_HEIGHT = 18  # Reduced from 25 to prevent over-merging rows
# High density threshold - rows with this many items are automatically valid anchors
HIGH_DENSITY_ITEMS = 6
# Minimum columns for consecutive row check
MIN_CONSECUTIVE_COLS = 4


def _analyze_row_structure(row_texts: List[List[float]], bbox_width: float) -> dict:
    """
    Analyze a row's structural properties.
    
    Returns dict with: num_items, num_cols, span_ratio, is_valid
    """
    result = {
        "num_items": len(row_texts),
        "num_cols": 0,
        "span_ratio": 0.0,
        "is_valid": False
    }
    
    if not row_texts:
        return result
    
    # Get unique X positions
    x_positions = [bbox[0] for bbox in row_texts]
    column_clusters = cluster_positions(x_positions, COLUMN_CLUSTER_TOLERANCE)
    result["num_cols"] = len(column_clusters)
    
    # Calculate span ratio
    if len(column_clusters) >= 2:
        col_centers = sorted([sum(c) / len(c) for c in column_clusters])
        col_span = col_centers[-1] - col_centers[0]
        result["span_ratio"] = col_span / bbox_width if bbox_width > 0 else 0
    
    # A row is structurally valid if it has:
    # 1. Enough items (at least MIN_ANCHOR_COLUMNS items)
    # 2. Enough columns (at least MIN_ANCHOR_COLUMNS columns)
    # 3. Columns spanning enough width (at least 60%)
    result["is_valid"] = (
        result["num_items"] >= MIN_ANCHOR_COLUMNS and  # Must have enough items!
        result["num_cols"] >= MIN_ANCHOR_COLUMNS and 
        result["span_ratio"] >= 0.60
    )
    
    return result


def refine_table_top_boundary(
    cluster_bbox: List[float], 
    text_elements: List[Dict]
) -> List[float]:
    """
    Refine the top boundary of a promoted table by trimming header rows
    until a row with proper table-like multi-column alignment is found.
    
    Vertical Continuity Rule:
    A row qualifies as an anchor if EITHER:
    1. HIGH DENSITY: The row has >= 6 items (real table headers are dense)
    2. CONSECUTIVE STRUCTURE: The row has >= 4 cols spanning >= 60% width
       AND the next row ALSO has >= 4 cols spanning >= 60% width
    
    This rejects single "one-hit wonder" rows like form metadata that 
    coincidentally have 4 items but aren't followed by similar structure.
    
    Args:
        cluster_bbox: Original cluster bounding box [x1, y1, x2, y2]
        text_elements: All text elements on the page
        
    Returns:
        Refined bounding box with corrected top boundary
    """
    if not cluster_bbox or len(cluster_bbox) != 4:
        return cluster_bbox
    
    x1, y1, x2, y2 = cluster_bbox
    bbox_width = x2 - x1
    
    if bbox_width <= 0:
        return cluster_bbox
    
    # Get text elements within this cluster
    overlapping_texts = []
    for text_elem in text_elements:
        text_bbox = text_elem.get("bbox", [])
        if text_bbox and bbox_overlap(cluster_bbox, text_bbox) >= 0.3:
            overlapping_texts.append(text_bbox)
    
    if len(overlapping_texts) < 6:
        return cluster_bbox
    
    # Group text elements into rows by Y position
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
    
    # Pre-analyze all rows
    row_analyses = [_analyze_row_structure(row, bbox_width) for row in rows]
    
    # Find the anchor using Vertical Continuity Rule
    anchor_y = y1  # Default to original top
    
    for i, (row_texts, analysis) in enumerate(zip(rows, row_analyses)):
        if not row_texts:
            continue
        
        # RULE 1: High density rows are automatic anchors
        # Real table headers have many items (e.g., NAME | STORES | NAME | STORES = 10+)
        if analysis["num_items"] >= HIGH_DENSITY_ITEMS:
            anchor_y = min(bbox[1] for bbox in row_texts)
            break
        
        # RULE 2: Consecutive structure - this row and next row both valid
        if analysis["is_valid"]:
            # Check if next row also has valid structure
            if i + 1 < len(row_analyses):
                next_analysis = row_analyses[i + 1]
                if next_analysis["is_valid"] or next_analysis["num_items"] >= HIGH_DENSITY_ITEMS:
                    # Two consecutive qualifying rows - use current as anchor
                    anchor_y = min(bbox[1] for bbox in row_texts)
                    break
            # Single valid row at end - still use it (might be last row of headers)
            elif i == len(rows) - 1:
                anchor_y = min(bbox[1] for bbox in row_texts)
                break
    
    # Only adjust if anchor is significantly different from original
    if anchor_y > y1 + ANCHOR_ROW_HEIGHT:
        return [x1, anchor_y, x2, y2]
    
    return cluster_bbox


def promote_layout_regions_to_tables(elements: List[Dict]) -> List[Dict]:
    """
    Promote clusters of layout_region elements to table elements.
    
    This heuristic detects borderless tables that the model missed by:
    1. Clustering vertically adjacent layout regions
    2. Validating each cluster for tabular structure
    3. Creating synthetic table elements for valid clusters
    
    Args:
        elements: List of element dictionaries
        
    Returns:
        Elements with promoted tables added
    """
    # Get existing tables
    existing_tables = [e for e in elements if e.get("type") == "table"]
    
    # Get text elements for validation
    text_elements = [e for e in elements if e.get("type") == "text"]
    
    # Cluster layout regions
    clusters = cluster_layout_regions_vertically(elements)
    
    promoted_tables = []
    
    for cluster in clusters:
        cluster_bbox = calculate_cluster_bbox(cluster)
        
        if not cluster_bbox:
            continue
        
        # Skip if this overlaps with an existing table
        if overlaps_existing_table(cluster_bbox, existing_tables):
            continue
        
        # Validate the cluster structure
        is_valid, score, signals = validate_cluster_as_table(cluster, text_elements)
        
        if is_valid:
            # Refine the top boundary to exclude single-column header rows
            # This addresses the issue where TO:, FROM:, SUBJECT: headers
            # get incorrectly included in the table bbox
            refined_bbox = refine_table_top_boundary(cluster_bbox, text_elements)
            
            # Create synthetic table element
            synthetic_table = {
                "type": "table",
                "bbox": refined_bbox,
                "confidence": 0.80,  # Lower confidence for heuristic detection
                "source": "heuristic_promotion",
                "structure_score": round(score, 1),
                "structure_signals": signals
            }
            promoted_tables.append(synthetic_table)
    
    # Add promoted tables to elements
    if promoted_tables:
        elements = elements + promoted_tables
    
    return elements


def postprocess_output(output_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply all post-processing to OCR output.
    
    Steps:
    0. Filter empty table/layout regions (remove hallucinations)
    0.5. Validate table structure (remove false positive tables)
    0.6. Heuristic table promotion (detect missed borderless tables)
    1. Deduplicate layout regions
    2. Score and flag potential hallucinations
    3. Clean text content
    4. Normalize phone numbers
    
    Args:
        output_data: Raw OCR output dictionary
        
    Returns:
        Cleaned output dictionary
    """
    for page in output_data.get("pages", []):
        elements = page.get("elements", [])
        
        # Step 0: Filter empty table/layout regions
        elements = filter_empty_regions(elements)
        
        # Step 0.5: Validate table structure (remove false positives)
        elements = filter_invalid_tables(elements)
        
        # Step 0.6: Heuristic table promotion (detect missed tables)
        elements = promote_layout_regions_to_tables(elements)
        
        # Step 1: Deduplicate layout regions
        elements = deduplicate_layout_regions(elements)
        
        # Step 2: Score and handle hallucinations
        elements = process_hallucinations(elements)
        
        # Step 3: Clean text content
        elements = clean_text_content(elements)
        
        # Step 4: Normalize phone numbers
        elements = normalize_phone_numbers(elements)
        
        page["elements"] = elements
    
    return output_data


def deduplicate_layout_regions(elements: List[Dict]) -> List[Dict]:
    """
    Remove duplicate layout_region elements with same bbox.
    Keeps one representative region per unique bbox.
    
    Args:
        elements: List of element dictionaries
        
    Returns:
        Deduplicated list of elements
    """
    seen_bboxes = set()
    deduplicated = []
    
    for element in elements:
        if element.get("type") == "layout_region":
            # Create a hashable key from bbox
            bbox = element.get("bbox", [])
            bbox_key = tuple(round(x, 2) for x in bbox) if bbox else ()
            
            if bbox_key in seen_bboxes:
                continue  # Skip duplicate
            seen_bboxes.add(bbox_key)
        
        deduplicated.append(element)
    
    return deduplicated


def process_hallucinations(elements: List[Dict]) -> List[Dict]:
    """
    Score each text element for hallucination likelihood using multiple signals.
    Removes high-confidence hallucinations, flags uncertain ones.
    
    Signals:
    - Confidence score (20%)
    - Text length (15%)
    - Character patterns (25%)
    - Bbox size anomaly (15%)
    - Dictionary check (15%)
    - Repetition patterns (10%)
    
    Args:
        elements: List of element dictionaries
        
    Returns:
        Processed elements with hallucinations handled
    """
    processed = []
    
    for element in elements:
        if element.get("type") != "text":
            processed.append(element)
            continue
        
        content = element.get("content", "")
        confidence = element.get("confidence", 1.0)
        bbox = element.get("bbox", [0, 0, 100, 100])
        
        # Calculate hallucination score
        score, signals = calculate_hallucination_score(content, confidence, bbox)
        
        if score > 0.70:
            # High hallucination likelihood - remove
            continue
        elif score > 0.40:
            # Uncertain - flag but keep
            element["hallucination_flag"] = True
            element["hallucination_score"] = round(score, 3)
            element["hallucination_signals"] = signals
        
        processed.append(element)
    
    return processed


def calculate_hallucination_score(
    content: str, 
    confidence: float, 
    bbox: List[float]
) -> Tuple[float, List[str]]:
    """
    Calculate hallucination likelihood score from multiple signals.
    
    Returns:
        Tuple of (score 0.0-1.0, list of triggered signal names)
    """
    score = 0.0
    signals = []
    
    # Signal 1: Low confidence (20% weight)
    if confidence < 0.50:
        score += 0.20
        signals.append("very_low_confidence")
    elif confidence < 0.70:
        score += 0.12
        signals.append("low_confidence")
    elif confidence < 0.85:
        score += 0.05
    
    # Signal 2: Very short text (15% weight)
    text_len = len(content.strip())
    if text_len <= 2:
        score += 0.15
        signals.append("very_short")
    elif text_len <= 4:
        score += 0.08
        signals.append("short")
    
    # Signal 3: Character pattern anomalies (25% weight)
    pattern_score, pattern_signals = check_character_patterns(content)
    score += pattern_score * 0.25
    signals.extend(pattern_signals)
    
    # Signal 4: Bbox size anomaly (15% weight)
    if len(bbox) >= 4:
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        
        # Very small bbox
        if width < 20 or height < 10:
            score += 0.15
            signals.append("tiny_bbox")
        # Extremely wide aspect ratio (likely noise)
        elif width > 0 and height / width > 5:
            score += 0.10
            signals.append("abnormal_aspect")
    
    # Signal 5: Not a recognizable word/pattern (15% weight)
    if not is_valid_text(content):
        score += 0.15
        signals.append("not_valid_text")
    
    # Signal 6: Repetition patterns (10% weight)
    if has_repetition_pattern(content):
        score += 0.10
        signals.append("repetition")
    
    return min(score, 1.0), signals


def check_character_patterns(content: str) -> Tuple[float, List[str]]:
    """
    Check for suspicious character patterns.
    
    Returns:
        Tuple of (score 0.0-1.0, list of pattern names)
    """
    score = 0.0
    patterns = []
    
    # All same character
    if len(set(content.replace(" ", ""))) == 1 and len(content) > 2:
        score += 0.8
        patterns.append("all_same_char")
    
    # Only digits when surrounded by text context is suspicious
    if content.strip().isdigit() and len(content.strip()) <= 3:
        score += 0.4
        patterns.append("isolated_digits")
    
    # Only punctuation
    if all(c in ".,;:!?-_'" for c in content.replace(" ", "")):
        score += 0.6
        patterns.append("only_punctuation")
    
    # Non-printable or unusual characters
    if any(ord(c) > 127 and c not in "éèêëàâäùûüôöîïç" for c in content):
        score += 0.3
        patterns.append("unusual_chars")
    
    # Mostly numbers with random letters (like "160", "000")
    if re.match(r'^[\d\s]+$', content.strip()) and len(content.strip()) <= 4:
        score += 0.5
        patterns.append("short_numbers")
    
    return min(score, 1.0), patterns


def is_valid_text(content: str) -> bool:
    """
    Check if content looks like valid text.
    Allows: words, numbers, dates, common abbreviations.
    """
    content = content.strip()
    
    # Empty
    if not content:
        return False
    
    # Single character (usually noise unless common)
    if len(content) == 1:
        return content.isalnum() or content in ".,;:!?()[]{}\"'"
    
    # Common patterns that are valid
    valid_patterns = [
        r'^[A-Za-z]{2,}$',  # Words
        r'^[A-Za-z]+[.,;:!?]?$',  # Words with punctuation
        r'^\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}$',  # Dates
        r'^\d+[.,]?\d*$',  # Numbers
        r'^[\$£€]\d+[.,]?\d*$',  # Currency
        r'^[A-Z]{2,}$',  # Acronyms
        r'^[A-Z][a-z]+$',  # Capitalized words
        r'^\(\d{3}\)\s*\d{3}[-\s]?\d{4}$',  # Phone numbers
        r'^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}$',  # Email
        r'^https?://',  # URLs
        r'^www\.',  # URLs
        r'^[A-Z][a-z]*\.?\s*$',  # Names with optional period
    ]
    
    for pattern in valid_patterns:
        if re.match(pattern, content, re.IGNORECASE):
            return True
    
    # Multi-word text is usually valid
    if ' ' in content and len(content) > 5:
        return True
    
    # Check if mostly alphanumeric
    alnum_ratio = sum(1 for c in content if c.isalnum()) / len(content)
    if alnum_ratio > 0.7:
        return True
    
    return False


def has_repetition_pattern(content: str) -> bool:
    """
    Check for repeated word patterns like 'the the the'.
    """
    words = content.lower().split()
    
    if len(words) < 2:
        return False
    
    # Check for consecutive repeated words
    for i in range(len(words) - 1):
        if words[i] == words[i + 1] and len(words[i]) > 1:
            return True
    
    # Check if all words are the same
    if len(set(words)) == 1 and len(words) > 2:
        return True
    
    return False


def clean_text_content(elements: List[Dict]) -> List[Dict]:
    """
    Clean and normalize text content.
    
    Fixes:
    - Extra whitespace
    - Common OCR substitution errors
    - Encoding issues
    """
    for element in elements:
        if element.get("type") != "text":
            continue
        
        content = element.get("content", "")
        
        # Normalize whitespace
        content = " ".join(content.split())
        
        # Fix common encoding issues
        content = fix_encoding_issues(content)
        
        element["content"] = content
    
    return elements


def fix_encoding_issues(text: str) -> str:
    """
    Fix common encoding/OCR issues.
    """
    replacements = {
        '\u00a0': ' ',  # Non-breaking space
        '\u2018': "'",  # Left single quote
        '\u2019': "'",  # Right single quote
        '\u201c': '"',  # Left double quote
        '\u201d': '"',  # Right double quote
        '\u2013': '-',  # En dash
        '\u2014': '-',  # Em dash
        '\u2026': '...',  # Ellipsis
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    return text


# ============================================================================
# Phone Number Normalization
# ============================================================================

# Comprehensive regex for US phone numbers - handles all 7 format variations
# Matches: (XXX) XXX-XXXX, XXX-XXX-XXXX, XXX XXX XXXX, XXX/XXX-XXXX, etc.
PHONE_PATTERN = re.compile(
    r'''
    (?:^|(?<=\s)|(?<=[:\-/]))  # Start of string, whitespace, or separator
    (?:
        \((\d{3})\)\s*         # (XXX) with optional space
        |
        (\d{3})[\s\-/.]        # XXX followed by separator
    )
    (\d{3})                    # Exchange: XXX
    [\s\-.]?                   # Optional separator
    (\d{4})                    # Subscriber: XXXX
    (?=\s|$|[,;])              # End boundary
    ''',
    re.VERBOSE
)

# Pattern to detect phone-related labels for type classification
PHONE_TYPE_PATTERNS = {
    'fax': re.compile(r'\b(FAX|FACSIMILE|TELECOPY)\b', re.IGNORECASE),
    'phone': re.compile(r'\b(PHONE|TELEPHONE|TEL|CALL)\b', re.IGNORECASE),
}

# False positive patterns to skip
ZIP_PATTERN = re.compile(r'\b\d{5}(-\d{4})?\b')  # 5-digit or 5+4 ZIP
DATE_PATTERN = re.compile(r'\b\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b')  # MM/DD/YY


def normalize_phone_numbers(elements: List[Dict]) -> List[Dict]:
    """
    Detect and normalize phone numbers in text elements.
    
    Adds:
    - normalized_phone: standardized format (XXX) XXX-XXXX
    - phone_type: 'phone' or 'fax' based on context
    
    False positive protection:
    - Skips ZIP codes (5-digit patterns)
    - Skips date patterns
    
    Args:
        elements: List of element dictionaries
        
    Returns:
        Elements with phone normalization applied
    """
    for element in elements:
        if element.get("type") != "text":
            continue
        
        content = element.get("content", "")
        
        # Skip if content looks like a date or ZIP code
        if is_date_or_zip(content):
            continue
        
        # Try to extract and normalize phone numbers
        phones = extract_phone_numbers(content)
        
        if phones:
            # Add normalized phones (multiple if found)
            if len(phones) == 1:
                element["normalized_phone"] = phones[0]
            else:
                element["normalized_phones"] = phones
            
            # Detect phone type from context
            phone_type = detect_phone_type(content)
            if phone_type:
                element["phone_type"] = phone_type
    
    return elements


def extract_phone_numbers(content: str) -> List[str]:
    """
    Extract and normalize all phone numbers from content.
    
    Returns:
        List of normalized phone numbers in (XXX) XXX-XXXX format
    """
    phones = []
    
    # First try the comprehensive regex
    for match in PHONE_PATTERN.finditer(content):
        # Get area code from either group 1 (parenthesized) or group 2 (plain)
        area_code = match.group(1) or match.group(2)
        exchange = match.group(3)
        subscriber = match.group(4)
        
        if area_code and exchange and subscriber:
            # Validate: must be 10 digits total
            if len(area_code) == 3 and len(exchange) == 3 and len(subscriber) == 4:
                normalized = f"({area_code}) {exchange}-{subscriber}"
                phones.append(normalized)
    
    # Also try simpler pattern for edge cases like "212/545-3297"
    slash_pattern = re.compile(r'(\d{3})/(\d{3})-(\d{4})')
    for match in slash_pattern.finditer(content):
        normalized = f"({match.group(1)}) {match.group(2)}-{match.group(3)}"
        if normalized not in phones:
            phones.append(normalized)
    
    # Handle space-separated format: "206 623 0594"
    space_pattern = re.compile(r'\b(\d{3})\s+(\d{3})\s+(\d{4})\b')
    for match in space_pattern.finditer(content):
        # Check it's not part of a date
        full_match = match.group(0)
        if not re.search(r'\d{1,2}[/\-]', content[max(0, match.start()-5):match.start()]):
            normalized = f"({match.group(1)}) {match.group(2)}-{match.group(3)}"
            if normalized not in phones:
                phones.append(normalized)
    
    return phones


def is_date_or_zip(content: str) -> bool:
    """
    Check if content is primarily a date or ZIP code (false positive).
    
    Returns:
        True if content should be skipped for phone normalization
    """
    content_stripped = content.strip()
    
    # Pure ZIP code pattern: "64105-2118" or "43215"
    if re.match(r'^\d{5}(-\d{4})?$', content_stripped):
        return True
    
    # Pure date pattern: "12/12/96", "01-14-99"
    if re.match(r'^\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}$', content_stripped):
        return True
    
    # Content is mostly a date with timestamp: "12/12/96 08:33"
    if re.match(r'^\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\s+\d{1,2}:\d{2}', content_stripped):
        return True
    
    # ZIP in address context: "City, State 12345-6789"
    # These should be kept as-is without phone normalization
    if re.search(r'[A-Za-z]+,?\s+[A-Za-z]{2}\s+\d{5}(-\d{4})?$', content_stripped):
        return True
    
    return False


def detect_phone_type(content: str) -> str:
    """
    Detect phone type (phone or fax) from content context.
    
    Returns:
        'phone', 'fax', or empty string if not determinable
    """
    content_upper = content.upper()
    
    # Check for fax-related keywords first (more specific)
    if PHONE_TYPE_PATTERNS['fax'].search(content):
        return 'fax'
    
    # Check for phone-related keywords
    if PHONE_TYPE_PATTERNS['phone'].search(content):
        return 'phone'
    
    return ""

