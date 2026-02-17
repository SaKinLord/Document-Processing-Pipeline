"""
Text processing utilities: text row clustering for reading order.
"""

import logging

logger = logging.getLogger(__name__)


def cluster_text_rows(elements, y_threshold=15):
    """
    Clusters text elements into rows based on Y-coordinate alignment.
    Sorts rows by Y, and elements within rows by X.
    Expected 'elements' is a list of dicts, each having a 'bbox' [x1, y1, x2, y2].
    """
    if not elements:
        return []

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
