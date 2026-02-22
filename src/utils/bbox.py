"""
Bounding box utilities: padding, size checks, word-level splitting, overlap calculations.
"""

from typing import Dict, List, Optional, Tuple

# Page dimension fallbacks (US Letter in PDF points, used only when no image is available)
DEFAULT_PAGE_WIDTH = 612
DEFAULT_PAGE_HEIGHT = 792


def estimate_page_dimensions(
    elements: List[Dict],
    page_dimensions: Optional[Tuple[int, int]] = None,
) -> Tuple[float, float]:
    """Estimate page width and height from element bboxes and optional image dimensions.

    Uses the maximum x2/y2 coordinates across all element bboxes, then takes
    the per-axis max with the actual image dimensions when provided.  Falls
    back to US Letter PDF points (612x792) if neither source yields a value.

    Args:
        elements: List of element dicts with optional ``bbox`` keys.
        page_dimensions: Optional ``(width, height)`` from the source image.

    Returns:
        ``(page_width, page_height)`` as floats.
    """
    page_width = 0.0
    page_height = 0.0
    for element in elements:
        bbox = element.get("bbox", [])
        if len(bbox) >= 4:
            page_width = max(page_width, bbox[2])
            page_height = max(page_height, bbox[3])

    if page_dimensions is not None:
        page_width = max(page_width, page_dimensions[0])
        page_height = max(page_height, page_dimensions[1])

    if page_width == 0:
        page_width = DEFAULT_PAGE_WIDTH
    if page_height == 0:
        page_height = DEFAULT_PAGE_HEIGHT

    return page_width, page_height


# ============================================================================
# Bbox Overlap Functions (consolidated from helpers.py + processing_pipeline.py)
# ============================================================================

def bbox_overlap_ratio_of_smaller(bbox1: List[float], bbox2: List[float]) -> float:
    """
    Calculate overlap ratio relative to the smaller bbox.

    Args:
        bbox1: First bbox [x1, y1, x2, y2]
        bbox2: Second bbox [x1, y1, x2, y2]

    Returns:
        Overlap ratio (0.0 to 1.0) relative to the smaller bbox
    """
    if len(bbox1) != 4 or len(bbox2) != 4:
        return 0.0

    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    intersection_area = (x2 - x1) * (y2 - y1)

    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    if area1 <= 0 or area2 <= 0:
        return 0.0

    smaller_area = min(area1, area2)
    return intersection_area / smaller_area


def bbox_overlap_ratio_of(bbox1: List[float], bbox2: List[float],
                          reference_bbox: List[float]) -> float:
    """
    Calculate overlap ratio relative to a specific reference bbox's area.

    Args:
        bbox1: First bbox [x1, y1, x2, y2]
        bbox2: Second bbox [x1, y1, x2, y2]
        reference_bbox: The bbox whose area is used as denominator

    Returns:
        Overlap ratio (0.0 to 1.0) relative to the reference bbox
    """
    if len(bbox1) != 4 or len(bbox2) != 4 or len(reference_bbox) != 4:
        return 0.0

    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    intersection_area = (x2 - x1) * (y2 - y1)

    ref_area = (reference_bbox[2] - reference_bbox[0]) * (reference_bbox[3] - reference_bbox[1])
    if ref_area <= 0:
        return 0.0

    return intersection_area / ref_area


def bboxes_intersect(bbox1: List[float], bbox2: List[float]) -> bool:
    """
    Check if two bounding boxes have any overlap at all.

    Args:
        bbox1: First bbox [x1, y1, x2, y2]
        bbox2: Second bbox [x1, y1, x2, y2]

    Returns:
        True if the bboxes overlap
    """
    if len(bbox1) != 4 or len(bbox2) != 4:
        return False

    return (bbox1[0] < bbox2[2] and bbox1[2] > bbox2[0] and
            bbox1[1] < bbox2[3] and bbox1[3] > bbox2[1])


def split_line_bbox_to_words(line_bbox, words, min_word_width=10):
    """
    Split a line-level bounding box into word-level bboxes.

    Distributes the line width proportionally based on word lengths (character count).
    This is an approximation since we don't have character-level metrics, but it's
    significantly better than using identical bboxes for all words in a line.

    Args:
        line_bbox: [x1, y1, x2, y2] - the line's bounding box
        words: List of word strings in the line
        min_word_width: Minimum width for each word bbox

    Returns:
        List of [x1, y1, x2, y2] bboxes, one per word
    """
    if not words:
        return []

    if len(words) == 1:
        return [line_bbox]

    x1, y1, x2, y2 = line_bbox
    line_width = x2 - x1

    word_lengths = [max(len(w), 1) for w in words]
    total_chars = sum(word_lengths)

    space_count = len(words) - 1
    total_chars_with_spacing = total_chars + (space_count * 0.5)

    if total_chars_with_spacing <= 0:
        word_width = line_width / len(words)
        return [[x1 + i * word_width, y1, x1 + (i + 1) * word_width, y2]
                for i in range(len(words))]

    width_per_char = line_width / total_chars_with_spacing

    word_bboxes = []
    current_x = x1

    for i, (word, word_len) in enumerate(zip(words, word_lengths)):
        word_width = max(word_len * width_per_char, min_word_width)
        word_x2 = min(current_x + word_width, x2)
        word_bboxes.append([current_x, y1, word_x2, y2])
        current_x = word_x2 + (0.5 * width_per_char)

    return word_bboxes


# Maximum page-area ratio allowed per visual element type.
# Detections covering more than this fraction of the page are likely
# full-page false positives from Florence-2 object detection.
BBOX_AREA_THRESHOLDS = {
    "human face": 0.10,
    "signature": 0.25,
    "logo": 0.30,
    "graphic": 0.50,
    "default": 0.50,
}


def is_bbox_too_large(bbox, width, height, label=None):
    """
    Checks if a bounding box covers too much of the page area.
    Uses per-type thresholds from BBOX_AREA_THRESHOLDS.
    """
    x1, y1, x2, y2 = bbox
    area = (x2 - x1) * (y2 - y1)

    total_area = width * height
    if total_area == 0:
        return False

    ratio = area / total_area

    threshold = BBOX_AREA_THRESHOLDS.get(label.lower(), BBOX_AREA_THRESHOLDS["default"]) if label else BBOX_AREA_THRESHOLDS["default"]

    return ratio > threshold
