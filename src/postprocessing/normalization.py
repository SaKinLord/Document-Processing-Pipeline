"""
Text normalization functions for OCR output.

Handles underscore field normalization, punctuation spacing, text cleaning,
TrOCR trailing period removal, parenthesis repair, and duplicate word removal.
"""

import re
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

_MARKUP_TAG_RE = re.compile(r'<[^>]+>')


def strip_markup_tags(text: str) -> str:
    """Strip HTML/math/formatting tags from OCR text output."""
    return _MARKUP_TAG_RE.sub('', text)


def normalize_underscores(text: str) -> str:
    """
    Normalize underscore fill-in patterns in a text string.

    Canonical text→text function shared by the pipeline and test framework.

    - '_ _ _ _' → '___'
    - '________' → '___'
    - 'Name:      ' → 'Name: ___'
    - Collapses double spaces

    Args:
        text: Raw text string

    Returns:
        Text with normalized underscores
    """
    # Normalize spaced underscores: '_ _ _ _' -> '____' then collapse
    text = re.sub(r'(_\s+){2,}_', '___', text)
    # Collapse runs of 3+ underscores to a single '___'
    text = re.sub(r'_{3,}', '___', text)
    # Collapse runs of 3+ spaces after a colon/label to ' ___'
    text = re.sub(r'(:\s*)\s{3,}', r'\1___', text)
    # Clean up any resulting double spaces
    text = re.sub(r' {2,}', ' ', text)
    return text


def normalize_underscore_fields(elements: List[Dict]) -> List[Dict]:
    """
    Normalize blank form fields (underscore runs, excessive spaces after labels).

    Applies normalize_underscores() to each text element.

    Args:
        elements: List of element dictionaries

    Returns:
        Elements with normalized underscore fields
    """
    for element in elements:
        if element.get("type") != "text":
            continue

        content = element.get("content", "")
        if not content:
            continue

        element["content"] = normalize_underscores(content).strip()

    return elements


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

        # Fix common encoding issues
        content = fix_encoding_issues(content)

        # Strip HTML/math markup tags (Surya OCR artifact)
        content = strip_markup_tags(content)

        # Normalize whitespace (re-normalize after stripping)
        content = " ".join(content.split())

        element["content"] = content

    return elements


def fix_encoding_issues(text: str) -> str:
    """
    Fix common encoding/OCR issues.
    """
    replacements = {
        '\u00a0': ' ',   # Non-breaking space
        '\u2018': "'",   # Left single quote
        '\u2019': "'",   # Right single quote
        '\u201c': '"',   # Left double quote
        '\u201d': '"',   # Right double quote
        '\u2013': '-',   # En dash
        '\u2014': '-',   # Em dash
        '\u2026': '...',  # Ellipsis
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text


# ============================================================================
# TrOCR Trailing Period Cleanup
# ============================================================================

# Abbreviations where a trailing period is legitimate and should be kept
TROCR_PERIOD_SAFE_ABBREVIATIONS = {
    'inc', 'corp', 'co', 'jr', 'sr', 'dr', 'mr', 'mrs', 'ms',
    'ltd', 'ave', 'st', 'blvd', 'dept', 'no', 'vs', 'etc',
}


def _clean_trocr_trailing_period(text: str) -> str:
    """Remove spurious trailing period from TrOCR output.

    TrOCR (trained on handwritten sentences) appends trailing periods to
    typed labels, headers, names, and numbers. This strips them while
    preserving legitimate abbreviation periods (Inc., Corp., Co., etc.).

    Also handles the " ." (space+period) pattern from TrOCR crop re-OCR.
    """
    text = text.rstrip()

    # Handle " ." pattern (space + period, common in short crop re-OCR)
    if text.endswith(' .'):
        return text[:-2].rstrip()

    # Handle trailing "."
    if text.endswith('.'):
        without_period = text[:-1]
        words = without_period.split()
        if words:
            last_word = words[-1].lower()
            # Protect known abbreviations
            if last_word in TROCR_PERIOD_SAFE_ABBREVIATIONS:
                return text
            # Protect single-letter abbreviations (e.g., "S." in initials)
            if len(last_word) == 1 and last_word.isalpha():
                return text
        return without_period

    return text


def strip_trocr_trailing_periods(elements: List[Dict], document_type: str = "typed") -> List[Dict]:
    """Strip spurious trailing periods from TrOCR text elements.

    TrOCR appends trailing periods when processing typed text (labels,
    headers, names, numbers). This is a model artifact from training on
    handwritten sentences that naturally end with periods.

    Skipped for handwritten documents where periods are legitimate
    end-of-sentence punctuation.

    Args:
        elements: List of element dictionaries
        document_type: Page classification ("typed", "handwritten", "mixed")

    Returns:
        Elements with spurious trailing periods removed
    """
    if document_type == "handwritten":
        return elements

    for element in elements:
        if element.get("type") != "text":
            continue
        if element.get("source_model") != "trocr":
            continue

        content = element.get("content", "")
        if not content:
            continue

        cleaned = _clean_trocr_trailing_period(content)
        if cleaned != content:
            logger.debug("TrOCR period stripped: '%s' → '%s'", content[:60], cleaned[:60])
            element["content"] = cleaned

    return elements


# Pattern to fix dropped opening parentheses for optional-plural suffixes.
# Surya OCR produces "BRANDS)" instead of "BRAND(S)", "DIVISIONS)" instead of "DIVISION(S)".
# Only matches known suffix patterns: S, s, ES, es (plural/optional markers).
_DROPPED_OPEN_PAREN_RE = re.compile(
    r'\b([A-Za-z]{2,}?)((?:[Ee][Ss]|[Ss]))\)', re.UNICODE
)


def repair_dropped_parentheses(text: str) -> str:
    """
    Repair dropped opening parentheses in OCR output.

    Surya OCR consistently drops the opening parenthesis in mid-word
    positions for optional-plural suffixes:
    - "BRANDS)" -> "BRAND(S)"
    - "DIVISIONS)" -> "DIVISION(S)"
    - "recipients)" -> "recipient(s):"
    - "individuals)" -> "individual(s)"

    Only repairs known suffix patterns (S), (s), (ES), (es) to avoid
    false positives on words like "SCOPE)" which should stay as-is.

    Args:
        text: OCR text to repair

    Returns:
        Text with parentheses repaired
    """
    if not text or ')' not in text:
        return text

    def fix_paren(match):
        prefix = match.group(1)
        suffix = match.group(2)
        return f"{prefix}({suffix})"

    # Only apply if there's a ) without a matching (
    open_count = text.count('(')
    close_count = text.count(')')

    if close_count > open_count:
        text = _DROPPED_OPEN_PAREN_RE.sub(fix_paren, text)

    return text


# ============================================================================
# Decimal-Dash Confusion Repair
# ============================================================================

# Surya OCR systematically misreads leading decimal points as dashes:
#   '.88' → '-88',  '.474' → '-474',  '.841' → '-841'
# Also misreads '.' as bullet '•' in numeric contexts:
#   '.55' → '•55'
#
# These patterns are detected by page-level context: if the page contains
# other decimal numbers (e.g., "14.00", "12.5"), dash-digit tokens are
# likely misread decimals rather than negative numbers.

# Standalone: space/start + dash + 2-4 digits + space/end/punct
_STANDALONE_DASH_DECIMAL_RE = re.compile(
    r'(?:^|(?<=\s))-(\d{2,4})(?=\s|$|[,;:!?\)\]])'
)
# Label-attached: ALL_CAPS_LABEL + dash + 2-4 digits (MENT-474 → MENT.474)
_LABEL_DASH_DECIMAL_RE = re.compile(
    r'([A-Z]{2,})-(\d{2,4})(?!\d)(?![-/a-zA-Z])'
)
# Bullet-decimal: •digit → .digit
_BULLET_DECIMAL_RE = re.compile(r'\u2022(\d)')
# Page-level decimal context detection
_DECIMAL_NUMBER_RE = re.compile(r'\b\d+\.\d+\b')
# Date/phone patterns to skip
_DATE_PATTERN_RE = re.compile(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}')
_PHONE_PATTERN_RE = re.compile(r'\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}')


def repair_decimal_dash_confusion(elements: List[Dict]) -> List[Dict]:
    """
    Fix Surya OCR confusion where decimal points are read as dashes or bullets.

    Uses page-level context: only corrects when the page already contains
    decimal numbers (e.g., "14.00", "12.5"), indicating a numeric/scientific
    context where fractional values are expected.

    Safe guards:
    - Requires 2-4 digits after the dash (skips '-5' which could be negative)
    - Skips date patterns (MM/DD/YYYY) and phone numbers
    - Standalone pattern only matches when dash is preceded by whitespace/start
    - Label pattern only matches ALL-CAPS labels (MENT-474, not semi-detached)

    Args:
        elements: List of element dictionaries

    Returns:
        Elements with decimal-dash confusions repaired
    """
    # Phase 1: Detect decimal context on the page
    all_text = ' '.join(
        e.get('content', '') for e in elements
        if e.get('type') == 'text' and e.get('content')
    )
    if not _DECIMAL_NUMBER_RE.search(all_text):
        return elements  # No decimal numbers on page → skip

    # Phase 2: Apply corrections element by element
    corrections = 0
    for element in elements:
        if element.get('type') != 'text':
            continue
        content = element.get('content', '')
        if not content:
            continue

        original = content

        # Always fix bullet-decimal (encoding issue, not context-dependent)
        content = _BULLET_DECIMAL_RE.sub(r'.\1', content)

        # Skip elements that are clearly dates or phone numbers
        if _DATE_PATTERN_RE.search(content):
            if content != original:
                element['content'] = content
            continue
        if _PHONE_PATTERN_RE.search(content):
            if content != original:
                element['content'] = content
            continue

        # Fix standalone dash-decimal tokens
        content = _STANDALONE_DASH_DECIMAL_RE.sub(r'.\1', content)

        # Fix label-attached dash-decimal (MENT-474 → MENT.474)
        content = _LABEL_DASH_DECIMAL_RE.sub(r'\1.\2', content)

        if content != original:
            element['content'] = content
            element['decimal_dash_repaired'] = True
            corrections += 1
            logger.debug("Decimal-dash repair: '%s' → '%s'",
                         original[:60], content[:60])

    if corrections:
        logger.info("  [decimal_dash_repair] corrected %d elements", corrections)

    return elements


def remove_consecutive_duplicate_words(elements: List[Dict]) -> List[Dict]:
    """
    Remove consecutive duplicate words from text content.

    This fixes a known TrOCR beam search artifact where words get duplicated,
    e.g., "We went straight straight to bed" -> "We went straight to bed"

    Args:
        elements: List of element dictionaries

    Returns:
        Elements with duplicate words removed from text content
    """
    for element in elements:
        if element.get("type") != "text":
            continue

        # Only apply to TrOCR output (handwriting model)
        if element.get("source_model") != "trocr":
            continue

        content = element.get("content", "")
        if not content:
            continue

        words = content.split()
        if len(words) < 2:
            continue

        # Remove consecutive duplicates (case-insensitive comparison)
        deduplicated = [words[0]]
        for word in words[1:]:
            if word.lower() != deduplicated[-1].lower():
                deduplicated.append(word)

        if len(deduplicated) < len(words):
            logger.debug("Duplicate words removed (%d): '%s' → '%s'",
                         len(words) - len(deduplicated), content[:60], ' '.join(deduplicated)[:60])
            element["content"] = ' '.join(deduplicated)
            element["duplicate_words_removed"] = len(words) - len(deduplicated)

    return elements
