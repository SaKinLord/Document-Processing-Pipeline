"""
Shared helper functions used across postprocessing submodules.
"""

from typing import List


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
