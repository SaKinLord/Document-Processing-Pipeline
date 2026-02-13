"""
Bounding box utilities: padding, size checks, word-level splitting.
"""


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


def is_bbox_too_large(bbox, width, height, label=None):
    """
    Checks if a bounding box covers too much of the page area.
    Uses per-type thresholds.
    """
    x1, y1, x2, y2 = bbox
    area = (x2 - x1) * (y2 - y1)

    total_area = width * height
    if total_area == 0:
        return False

    ratio = area / total_area

    AREA_THRESHOLDS = {
        "human face": 0.10,
        "signature": 0.25,
        "logo": 0.30,
        "graphic": 0.50,
        "default": 0.50
    }

    threshold = AREA_THRESHOLDS.get(label.lower(), AREA_THRESHOLDS["default"]) if label else AREA_THRESHOLDS["default"]

    return ratio > threshold
