"""
Hallucination detection and filtering for OCR output.

Multi-signal scoring system to identify and remove hallucinated text elements,
rotated margin text fragments, and other OCR artifacts.
"""

import re
import logging
from typing import Dict, List, Optional, Tuple

from src.utils.bbox import estimate_page_dimensions, DEFAULT_PAGE_WIDTH, DEFAULT_PAGE_HEIGHT

logger = logging.getLogger(__name__)

# ============================================================================
# Decision thresholds
# ============================================================================
HALLUCINATION_REMOVE_THRESHOLD = 0.50   # Score >= this → remove element
HALLUCINATION_FLAG_THRESHOLD = 0.30     # Score > this → flag but keep

# ============================================================================
# Signal 1: Confidence (15% max weight)
# ============================================================================
CONFIDENCE_VERY_LOW = 0.50              # Below this → very_low_confidence
CONFIDENCE_VERY_LOW_WEIGHT = 0.15
CONFIDENCE_LOW = 0.70                   # Below this → low_confidence
CONFIDENCE_LOW_WEIGHT = 0.10
CONFIDENCE_MODERATE = 0.85              # Below this → slight penalty
CONFIDENCE_MODERATE_WEIGHT = 0.05

# ============================================================================
# Signal 2: Text length (10% max weight)
# ============================================================================
TEXT_VERY_SHORT_MAX = 2                 # <= this many chars → very_short
TEXT_VERY_SHORT_WEIGHT = 0.10
TEXT_SHORT_MAX = 4                      # <= this many chars → short
TEXT_SHORT_WEIGHT = 0.06

# ============================================================================
# Signal 3: Character patterns (25% max weight, applied as multiplier)
# ============================================================================
CHAR_PATTERN_WEIGHT = 0.25
ALL_SAME_CHAR_SCORE = 0.8              # All identical characters
ISOLATED_DIGITS_MAX_LEN = 3            # Max length for isolated_digits signal
ISOLATED_DIGITS_SCORE = 0.30
SHORT_NUMBERS_MAX_LEN = 4              # Max length for short_numbers signal
SHORT_NUMBERS_SCORE = 0.35
PUNCTUATION_ONLY_SCORE = 0.6           # All punctuation characters
UNUSUAL_CHARS_SCORE = 0.3              # Non-ASCII outside Latin Extended
ISOLATED_YEAR_SCORE = 0.25             # Lone year (e.g. "1998")

# ============================================================================
# Signal 4: Bbox anomaly (15% max weight)
# ============================================================================
BBOX_MIN_WIDTH = 20                    # Pixels — below this → tiny_bbox
BBOX_MIN_HEIGHT = 10                   # Pixels — below this → tiny_bbox
BBOX_TINY_WEIGHT = 0.15
BBOX_ASPECT_RATIO_MAX = 5              # height/width — above this → abnormal
BBOX_ABNORMAL_ASPECT_WEIGHT = 0.10

# ============================================================================
# Signal 5: Valid text check (15% weight)
# ============================================================================
INVALID_TEXT_WEIGHT = 0.15
VALID_TEXT_MULTIWORD_MIN_LEN = 5       # Multi-word text longer than this → valid
VALID_TEXT_ALNUM_RATIO = 0.7           # Alphanumeric ratio above this → valid

# ============================================================================
# Signal 6: Repetition (10% weight)
# ============================================================================
REPETITION_WEIGHT = 0.10

# ============================================================================
# Signal 7: Margin position (10-25% weight depending on length)
# ============================================================================
MARGIN_EDGE_THRESHOLD = 0.03           # 3% of page width — outer edge
MARGIN_INNER_THRESHOLD = 0.10          # 10% of page width — inner boundary
MARGIN_SHORT_FRAGMENT_WEIGHT = 0.25    # Short non-numeric margin fragment
MARGIN_LONG_WEIGHT = 0.10             # Longer text at margin

# ============================================================================
# Rotated margin text filter
# ============================================================================
ROTATED_RIGHT_EDGE_THRESHOLD = 0.92    # Text starting past 92% of page width
ROTATED_NARROW_THRESHOLD = 0.04        # Max 4% of page width for narrow bbox


def process_hallucinations(elements: List[Dict],
                           page_dimensions: Optional[Tuple[int, int]] = None) -> List[Dict]:
    """
    Score each text element for hallucination likelihood using multiple signals.
    Removes high-confidence hallucinations, flags uncertain ones.

    Signals:
    - Confidence score (15%)
    - Text length (10%)
    - Character patterns (25%)
    - Bbox size anomaly (15%)
    - Dictionary check (15%)
    - Repetition patterns (10%)
    - Margin position (10-25%)

    Args:
        elements: List of element dictionaries
        page_dimensions: Optional (width, height) from the source image.

    Returns:
        Processed elements with hallucinations handled
    """
    page_width, page_height = estimate_page_dimensions(elements, page_dimensions)

    processed = []
    removed_count = 0
    flagged_count = 0

    for element in elements:
        if element.get("type") != "text":
            processed.append(element)
            continue

        content = element.get("content", "")
        confidence = element.get("confidence", 1.0)
        bbox = element.get("bbox", [0, 0, 100, 100])

        # Calculate hallucination score
        score, signals = calculate_hallucination_score(
            content, confidence, bbox, page_width, page_height
        )

        if score >= HALLUCINATION_REMOVE_THRESHOLD:
            # High hallucination likelihood - remove
            logger.debug("Hallucination removed (score=%.3f): %s | signals=%s",
                         score, content[:60], signals)
            removed_count += 1
            continue
        elif score > HALLUCINATION_FLAG_THRESHOLD:
            # Uncertain - flag but keep
            element["hallucination_flag"] = True
            element["hallucination_score"] = round(score, 3)
            element["hallucination_signals"] = signals
            logger.debug("Hallucination flagged (score=%.3f): %s | signals=%s",
                         score, content[:60], signals)
            flagged_count += 1

        processed.append(element)

    if removed_count or flagged_count:
        logger.info("Hallucination filter: %d removed, %d flagged", removed_count, flagged_count)

    return processed


def filter_rotated_margin_text(elements: List[Dict],
                               page_dimensions: Optional[Tuple[int, int]] = None) -> List[Dict]:
    """
    Remove text fragments from rotated margin text (e.g., vertical Bates numbers).

    Documents often have ID numbers printed vertically along the right edge.
    The OCR reads these as isolated fragments (single digits, short nonsense
    strings) scattered along the margin. This filter detects and removes them
    based on their distinctive bbox signature: extremely narrow, at the far
    right edge of the page.

    Digit-only content is exempted -- Bates numbers and document IDs that Surya
    reads as a single coherent number are legitimate and should be kept.

    Args:
        elements: List of element dictionaries
        page_dimensions: Optional (width, height) from the source image

    Returns:
        Filtered elements with rotated margin fragments removed
    """
    page_width, _ = estimate_page_dimensions(elements, page_dimensions)
    if page_width <= 0:
        return elements  # Can't determine margins without bboxes

    filtered = []
    for element in elements:
        if element.get("type") != "text":
            filtered.append(element)
            continue

        bbox = element.get("bbox", [0, 0, 100, 100])
        if len(bbox) < 4:
            filtered.append(element)
            continue

        content = element.get("content", "").strip()
        bbox_width = bbox[2] - bbox[0]
        # Rotated margin text: starts in rightmost 8% of page AND very narrow
        at_right_edge = bbox[0] > page_width * ROTATED_RIGHT_EDGE_THRESHOLD
        very_narrow = bbox_width < page_width * ROTATED_NARROW_THRESHOLD

        if at_right_edge and very_narrow:
            # Exempt digit-only content (Bates numbers, document IDs)
            if content.isdigit():
                filtered.append(element)
                continue
            continue  # Drop rotated margin fragment

        filtered.append(element)

    return filtered


def calculate_hallucination_score(
    content: str,
    confidence: float,
    bbox: List[float],
    page_width: float = DEFAULT_PAGE_WIDTH,
    page_height: float = DEFAULT_PAGE_HEIGHT
) -> Tuple[float, List[str]]:
    """
    Calculate hallucination likelihood score from multiple signals.

    Returns:
        Tuple of (score 0.0-1.0, list of triggered signal names)
    """
    score = 0.0
    signals = []

    # Signal 1: Low confidence (15% max weight)
    if confidence < CONFIDENCE_VERY_LOW:
        score += CONFIDENCE_VERY_LOW_WEIGHT
        signals.append("very_low_confidence")
    elif confidence < CONFIDENCE_LOW:
        score += CONFIDENCE_LOW_WEIGHT
        signals.append("low_confidence")
    elif confidence < CONFIDENCE_MODERATE:
        score += CONFIDENCE_MODERATE_WEIGHT

    # Signal 2: Very short text (10% max weight)
    text_len = len(content.strip())
    if text_len <= TEXT_VERY_SHORT_MAX:
        score += TEXT_VERY_SHORT_WEIGHT
        signals.append("very_short")
    elif text_len <= TEXT_SHORT_MAX:
        score += TEXT_SHORT_WEIGHT
        signals.append("short")

    # Signal 3: Character pattern anomalies (25% max weight)
    pattern_score, pattern_signals = check_character_patterns(content)
    score += pattern_score * CHAR_PATTERN_WEIGHT
    signals.extend(pattern_signals)

    # Signal 4: Bbox size anomaly (15% max weight)
    if len(bbox) >= 4:
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]

        # Very small bbox
        if width < BBOX_MIN_WIDTH or height < BBOX_MIN_HEIGHT:
            score += BBOX_TINY_WEIGHT
            signals.append("tiny_bbox")
        # Extremely tall aspect ratio (likely noise)
        elif width > 0 and height / width > BBOX_ASPECT_RATIO_MAX:
            score += BBOX_ABNORMAL_ASPECT_WEIGHT
            signals.append("abnormal_aspect")

    # Signal 5: Not a recognizable word/pattern (15% weight)
    if not is_valid_text(content):
        score += INVALID_TEXT_WEIGHT
        signals.append("not_valid_text")

    # Signal 6: Repetition patterns (10% weight)
    if has_repetition_pattern(content):
        score += REPETITION_WEIGHT
        signals.append("repetition")

    # Signal 7: Margin position (10-25% weight depending on fragment length)
    if len(bbox) >= 4 and page_width > 0:
        at_left_margin = (bbox[0] < page_width * MARGIN_EDGE_THRESHOLD
                          and bbox[2] < page_width * MARGIN_INNER_THRESHOLD)
        at_right_margin = (bbox[0] > page_width * (1 - MARGIN_INNER_THRESHOLD)
                           and bbox[2] > page_width * (1 - MARGIN_EDGE_THRESHOLD))
        is_page_number = content.strip().isdigit()
        if (at_left_margin or at_right_margin) and not is_page_number:
            if text_len <= TEXT_SHORT_MAX:
                # Short non-numeric fragment at margin
                score += MARGIN_SHORT_FRAGMENT_WEIGHT
                signals.append("margin_fragment_short")
            else:
                # Longer text at margin
                score += MARGIN_LONG_WEIGHT
                signals.append("margin_position")

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
        score += ALL_SAME_CHAR_SCORE
        patterns.append("all_same_char")

    stripped = content.strip()

    # Only digits / short numbers -- mutually exclusive to avoid double-counting.
    if stripped.isdigit() and len(stripped) <= ISOLATED_DIGITS_MAX_LEN:
        score += ISOLATED_DIGITS_SCORE
        patterns.append("isolated_digits")
    elif re.match(r'^[\d\s]+$', stripped) and len(stripped) <= SHORT_NUMBERS_MAX_LEN:
        score += SHORT_NUMBERS_SCORE
        patterns.append("short_numbers")

    # Only punctuation
    if all(c in ".,;:!?-_'" for c in content.replace(" ", "")):
        score += PUNCTUATION_ONLY_SCORE
        patterns.append("only_punctuation")

    # Non-printable or unusual characters -- allow Latin Extended
    if any(
        ord(c) > 127
        and not (0x00C0 <= ord(c) <= 0x024F)
        and not (0x1E00 <= ord(c) <= 0x1EFF)
        for c in content
    ):
        score += UNUSUAL_CHARS_SCORE
        patterns.append("unusual_chars")

    # Isolated year patterns (common hallucinations from fine print)
    if re.match(r'^(19|20)\d{2}s?$', stripped):
        score += ISOLATED_YEAR_SCORE
        patterns.append("isolated_year")

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

    # Case-sensitive patterns — the case distinction is meaningful
    case_sensitive_patterns = [
        r'^[A-Z]{2,}$',                          # Acronyms (all uppercase)
        r'^[A-Z][a-z]+$',                        # Capitalized words
        r'^[A-Z][a-z]*\.?\s*$',                  # Names with optional period
    ]
    for pattern in case_sensitive_patterns:
        if re.match(pattern, content):
            return True

    # Case-insensitive patterns — case doesn't matter
    case_insensitive_patterns = [
        r'^[A-Za-z]{2,}$',                       # Words (any case)
        r'^[A-Za-z]+[.,;:!?]?$',                 # Words with punctuation
        r'^\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}$',  # Dates
        r'^\d+[.,]?\d*$',                        # Numbers
        r'^[\$\u00a3\u20ac]\d+[.,]?\d*$',        # Currency
        r'^\(\d{3}\)\s*\d{3}[-\s]?\d{4}$',       # Phone numbers
        r'^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$',  # Email
        r'^https?://',                            # URLs
        r'^www\.',                                # URLs
    ]
    for pattern in case_insensitive_patterns:
        if re.match(pattern, content, re.IGNORECASE):
            return True

    # Multi-word text is usually valid
    if ' ' in content and len(content) > VALID_TEXT_MULTIWORD_MIN_LEN:
        return True

    # Check if mostly alphanumeric
    alnum_ratio = sum(1 for c in content if c.isalnum()) / len(content)
    if alnum_ratio > VALID_TEXT_ALNUM_RATIO:
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
