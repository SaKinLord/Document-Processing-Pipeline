"""
Hallucination detection and filtering for OCR output.

Multi-signal scoring system to identify and remove hallucinated text elements,
rotated margin text fragments, and other OCR artifacts.
"""

import re
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


def process_hallucinations(elements: List[Dict]) -> List[Dict]:
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
    - Margin position (10%)

    Args:
        elements: List of element dictionaries

    Returns:
        Processed elements with hallucinations handled
    """
    # Estimate page dimensions from all element bboxes
    page_width = 0
    page_height = 0
    for element in elements:
        bbox = element.get("bbox", [])
        if len(bbox) >= 4:
            page_width = max(page_width, bbox[2])
            page_height = max(page_height, bbox[3])
    # Fallback if no bboxes found
    if page_width == 0:
        page_width = 612  # Standard letter width in points
    if page_height == 0:
        page_height = 792

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

        if score >= 0.50:
            # High hallucination likelihood - remove (tightened from > 0.50)
            logger.debug("Hallucination removed (score=%.3f): %s | signals=%s",
                         score, content[:60], signals)
            removed_count += 1
            continue
        elif score > 0.30:
            # Uncertain - flag but keep (tightened from 0.40)
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


def filter_rotated_margin_text(elements: List[Dict]) -> List[Dict]:
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

    Returns:
        Filtered elements with rotated margin fragments removed
    """
    # Estimate page width from all element bboxes
    page_width = 0
    for element in elements:
        bbox = element.get("bbox", [])
        if len(bbox) >= 4:
            page_width = max(page_width, bbox[2])
    if page_width == 0:
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
        at_right_edge = bbox[0] > page_width * 0.92
        very_narrow = bbox_width < page_width * 0.04

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
    page_width: float = 612,
    page_height: float = 792
) -> Tuple[float, List[str]]:
    """
    Calculate hallucination likelihood score from multiple signals.

    Returns:
        Tuple of (score 0.0-1.0, list of triggered signal names)
    """
    score = 0.0
    signals = []

    # Signal 1: Low confidence (15% weight)
    if confidence < 0.50:
        score += 0.15
        signals.append("very_low_confidence")
    elif confidence < 0.70:
        score += 0.10
        signals.append("low_confidence")
    elif confidence < 0.85:
        score += 0.05

    # Signal 2: Very short text (10% weight)
    text_len = len(content.strip())
    if text_len <= 2:
        score += 0.10
        signals.append("very_short")
    elif text_len <= 4:
        score += 0.06
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

    # Signal 7: Margin position (10% weight)
    if len(bbox) >= 4 and page_width > 0:
        margin_threshold = 0.03  # 3% of page dimension
        at_left_margin = bbox[0] < page_width * margin_threshold and bbox[2] < page_width * 0.10
        at_right_margin = bbox[0] > page_width * (1 - 0.10) and bbox[2] > page_width * (1 - margin_threshold)
        is_page_number = content.strip().isdigit()
        if (at_left_margin or at_right_margin) and not is_page_number:
            if text_len <= 4:
                # Short non-numeric fragment at margin
                score += 0.25
                signals.append("margin_fragment_short")
            else:
                # Longer text at margin
                score += 0.10
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
        score += 0.8
        patterns.append("all_same_char")

    stripped = content.strip()

    # Only digits / short numbers -- mutually exclusive to avoid double-counting.
    if stripped.isdigit() and len(stripped) <= 3:
        score += 0.30
        patterns.append("isolated_digits")
    elif re.match(r'^[\d\s]+$', stripped) and len(stripped) <= 4:
        score += 0.35
        patterns.append("short_numbers")

    # Only punctuation
    if all(c in ".,;:!?-_'" for c in content.replace(" ", "")):
        score += 0.6
        patterns.append("only_punctuation")

    # Non-printable or unusual characters -- allow Latin Extended
    if any(
        ord(c) > 127
        and not (0x00C0 <= ord(c) <= 0x024F)
        and not (0x1E00 <= ord(c) <= 0x1EFF)
        for c in content
    ):
        score += 0.3
        patterns.append("unusual_chars")

    # Isolated year patterns (common hallucinations from fine print)
    if re.match(r'^(19|20)\d{2}s?$', stripped):
        score += 0.25
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

    # Common patterns that are valid
    valid_patterns = [
        r'^[A-Za-z]{2,}$',  # Words
        r'^[A-Za-z]+[.,;:!?]?$',  # Words with punctuation
        r'^\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}$',  # Dates
        r'^\d+[.,]?\d*$',  # Numbers
        r'^[\$\u00a3\u20ac]\d+[.,]?\d*$',  # Currency
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
