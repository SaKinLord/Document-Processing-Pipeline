"""
Text processing utilities: text row clustering for reading order.
"""

import logging

logger = logging.getLogger(__name__)

# Default row clustering threshold as a ratio of page height.
# ~1.2% of page height ≈ 15px on a 1250px-tall image (typical 200 DPI scan).
DEFAULT_ROW_THRESHOLD_RATIO = 0.012
# Absolute minimum threshold in pixels to avoid over-clustering on tiny images.
MIN_ROW_THRESHOLD_PX = 8


def cluster_text_rows(elements, y_threshold=None):
    """
    Clusters text elements into rows based on Y-coordinate alignment.
    Sorts rows by Y, and elements within rows by X.
    Expected 'elements' is a list of dicts, each having a 'bbox' [x1, y1, x2, y2].

    The clustering threshold scales with page height so it behaves
    consistently across different scan resolutions (72 DPI, 200 DPI,
    300 DPI, etc.).  Pass an explicit ``y_threshold`` in pixels to
    override the adaptive calculation.
    """
    if not elements:
        return []

    # Adaptive threshold: derive from max y-coordinate across all elements
    if y_threshold is None:
        max_y = max(
            (e['bbox'][3] for e in elements if e.get('bbox') and len(e['bbox']) >= 4),
            default=0,
        )
        y_threshold = max(MIN_ROW_THRESHOLD_PX, max_y * DEFAULT_ROW_THRESHOLD_RATIO)

    sorted_elements = sorted(elements, key=lambda e: e['bbox'][1])

    rows = []
    current_row = []

    for elem in sorted_elements:
        if not current_row:
            current_row.append(elem)
            continue

        ref_elem = current_row[0]

        if abs(elem['bbox'][1] - ref_elem['bbox'][1]) < y_threshold:
            current_row.append(elem)
        else:
            current_row.sort(key=lambda e: e['bbox'][0])
            rows.append(current_row)
            current_row = [elem]

    if current_row:
        current_row.sort(key=lambda e: e['bbox'][0])
        rows.append(current_row)

    ordered_elements = []
    for row_idx, row in enumerate(rows):
        for elem in row:
            elem['row_id'] = row_idx
            ordered_elements.append(elem)

    return ordered_elements
