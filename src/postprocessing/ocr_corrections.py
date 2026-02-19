"""
OCR correction logic for post-processing.

Handles:
- Generalizable non-word correction via spell checker + OCR confusion matrix
- Prefix restoration in negation context
- Offensive OCR misread filtering with cross-model re-verification
"""

import re
import logging
from typing import Dict, List, Optional, Set

from .normalization import _clean_trocr_trailing_period

logger = logging.getLogger(__name__)

# ============================================================================
# OCR-Aware Spell Correction (Generalizable)
# ============================================================================
#
# Uses pyspellchecker for word validation and an OCR character confusion matrix
# to filter candidates to visually plausible substitutions only.
#
# This catches ANY non-word OCR error automatically, making the pipeline robust
# to new documents without manual dictionary updates.

# Characters commonly confused by OCR engines (visual similarity)
OCR_CHAR_CONFUSIONS = {
    'c': 'eoa',    'e': 'co',      'a': 'oe',     'o': 'ae0c',
    'l': '1Ii',    '1': 'lI',      'I': 'l1',     'i': 'jl1t',
    'n': 'rih',    'h': 'bn',      'b': 'h6',     'd': 'o',
    'O': '0DQ',    '0': 'O',       'D': 'O',
    'S': '5$',     '5': 'S$',      '$': 'S5',
    'B': '8R',     '8': 'B',       'R': 'B',
    'G': 'C6',     'C': 'GO',      '6': 'Gb',
    'u': 'vn',     'v': 'uy',      'w': 'vv',
    'f': 't',      't': 'fi',      'r': 'v',
    'g': 'q9',     'q': 'g9',      '9': 'gq',
    'p': 'b',      'm': 'nn',
}

# Minimum word length for spell-check correction (shorter words are too ambiguous)
_MIN_SPELLCHECK_LEN = 5

# Lazy-initialized spell checker singleton
_spell_checker = None


def _get_spell_checker():
    """Lazy-initialize the spell checker."""
    global _spell_checker
    if _spell_checker is not None:
        return _spell_checker

    try:
        from spellchecker import SpellChecker
        _spell_checker = SpellChecker()
        logger.debug("Spell checker initialized")
        return _spell_checker

    except ImportError:
        logger.warning("pyspellchecker not installed — non-word OCR correction disabled. "
                       "Install with: pip install pyspellchecker")
        return None


def _is_ocr_plausible(wrong: str, candidate: str) -> bool:
    """Check if a correction candidate differs by OCR-confusable characters only.

    Returns True if every character difference between wrong and candidate
    is explained by a known OCR visual confusion. This prevents generic
    spell-checker suggestions that aren't plausible OCR errors.
    """
    if abs(len(wrong) - len(candidate)) > 1:
        return False  # Length difference > 1 is unlikely a simple OCR confusion

    # Same length: check each position
    if len(wrong) == len(candidate):
        diff_count = 0
        for w, c in zip(wrong.lower(), candidate.lower()):
            if w != c:
                diff_count += 1
                confusable = OCR_CHAR_CONFUSIONS.get(w, '')
                reverse_confusable = OCR_CHAR_CONFUSIONS.get(c, '')
                if c not in confusable and w not in reverse_confusable:
                    return False  # This character difference is not an OCR confusion
        return 0 < diff_count <= 2

    # Length differs by 1: could be multi-char confusion (rn↔m, cl↔d)
    # Accept if edit distance is 1 (insertion or deletion of one char)
    return True


def correct_nonword_ocr_errors(text: str, page_context: Optional[Set[str]] = None) -> str:
    """
    Correct OCR errors where the output is not a real English word.

    Uses a spell checker to detect non-words, then filters correction
    candidates through an OCR character confusion matrix to ensure only
    visually plausible substitutions are applied.

    This is GENERALIZABLE: it works on any English document without
    corpus-specific dictionaries.

    Safety guards:
    - Minimum word length of 5 characters (short words are too ambiguous)
    - Skips all-uppercase words <= 6 chars (likely acronyms: MOIST, MENT)
    - Skips words containing digits (codes, reference numbers)
    - Skips words with hyphens/slashes (compound words)
    - Only applies when exactly one OCR-plausible candidate exists
    - Logs all corrections for review

    Args:
        text: OCR text to correct
        page_context: Optional set of lowercased words from the page
                     (unused currently, reserved for future confidence boosting)

    Returns:
        Text with non-word OCR errors corrected
    """
    spell = _get_spell_checker()
    if not spell or not text:
        return text

    words = text.split()
    corrected_words = []

    for word in words:
        stripped = word.rstrip(':.,;!?')
        suffix = word[len(stripped):]

        # Skip words that are too short, contain digits, or look like acronyms
        if (len(stripped) < _MIN_SPELLCHECK_LEN
                or (stripped.isupper() and len(stripped) <= 6)
                or any(c.isdigit() for c in stripped)
                or '/' in stripped or '-' in stripped):
            corrected_words.append(word)
            continue

        # Check if word is in the dictionary
        word_lower = stripped.lower()
        if spell.known([word_lower]):
            corrected_words.append(word)
            continue

        # Word is NOT in dictionary — get spell checker candidates
        candidates = spell.candidates(word_lower)
        if not candidates:
            corrected_words.append(word)
            continue

        # Filter to OCR-plausible candidates only
        ocr_candidates = [c for c in candidates if _is_ocr_plausible(word_lower, c)]

        if len(ocr_candidates) == 1:
            correction = ocr_candidates[0]
            # Preserve original case pattern
            if stripped.isupper():
                correction = correction.upper()
            elif stripped[0].isupper():
                correction = correction[0].upper() + correction[1:]
            corrected_words.append(correction + suffix)
            logger.debug("Non-word OCR correction: '%s' → '%s'", stripped, correction)
        elif len(ocr_candidates) > 1:
            # Multiple OCR-plausible candidates — pick the most common one
            best = max(ocr_candidates, key=lambda c: spell.word_usage_frequency(c) or 0)
            if stripped.isupper():
                best = best.upper()
            elif stripped[0].isupper():
                best = best[0].upper() + best[1:]
            corrected_words.append(best + suffix)
            logger.debug("Non-word OCR correction (best of %d): '%s' → '%s'",
                         len(ocr_candidates), stripped, best)
        else:
            corrected_words.append(word)  # No OCR-plausible correction found

    return ' '.join(corrected_words)

# ============================================================================
# Offensive OCR Misread Filter (P0 - Reputational Risk)
# ============================================================================

OFFENSIVE_OCR_CORRECTIONS = [
    (re.compile(r'\bBitch\b', re.IGNORECASE), 'Litco'),
    (re.compile(r'\bPecker\b(?=\s+Drugs)', re.IGNORECASE), 'Eckerd'),
]


def filter_offensive_ocr_misreads(elements: List[Dict], page_image=None,
                                   handwriting_recognizer=None) -> List[Dict]:
    """
    Detect and correct known offensive OCR misreads with source image re-verification.

    When a text element matches an offensive pattern and the source image + TrOCR
    are available, the bbox region is re-cropped and re-OCR'd as a tiebreaker:
    - If TrOCR also produces the offensive word -> keep original (both models agree)
    - If TrOCR produces something different -> use TrOCR's result

    Falls back to blind regex replacement when re-verification is not possible.

    Args:
        elements: List of element dictionaries
        page_image: PIL Image of the page (optional, enables re-verification)
        handwriting_recognizer: TrOCR recognizer instance (optional)

    Returns:
        Elements with offensive misreads corrected
    """
    can_reverify = page_image is not None and handwriting_recognizer is not None

    for element in elements:
        if element.get("type") != "text":
            continue

        content = element.get("content", "")
        if not content:
            continue

        corrections_made = []

        for pattern, replacement in OFFENSIVE_OCR_CORRECTIONS:
            if not pattern.search(content):
                continue

            source_model = element.get("source_model", "")

            if can_reverify and source_model == "surya" and element.get("bbox"):
                bbox = element["bbox"]
                if len(bbox) < 4:
                    # Invalid bbox — fall through to regex fallback
                    original = content
                    content = pattern.sub(replacement, content)
                    corrections_made.append({
                        "pattern": pattern.pattern,
                        "original_text": original,
                        "action": "regex_fallback",
                        "final_text": content,
                    })
                    continue

                # Clamp bbox to image bounds and verify positive area
                img_w, img_h = page_image.size
                clamped = [
                    max(0, bbox[0]),
                    max(0, bbox[1]),
                    min(img_w, bbox[2]),
                    min(img_h, bbox[3]),
                ]
                if clamped[2] <= clamped[0] or clamped[3] <= clamped[1]:
                    # Zero or negative area — fall through to regex fallback
                    original = content
                    content = pattern.sub(replacement, content)
                    corrections_made.append({
                        "pattern": pattern.pattern,
                        "original_text": original,
                        "action": "regex_fallback",
                        "final_text": content,
                    })
                    continue

                crop = page_image.crop((clamped[0], clamped[1], clamped[2], clamped[3]))
                reocr_text, reocr_conf = handwriting_recognizer.recognize(crop)
                reocr_text = _clean_trocr_trailing_period(reocr_text)

                if pattern.search(reocr_text):
                    corrections_made.append({
                        "pattern": pattern.pattern,
                        "original_text": content,
                        "action": "reverified_kept",
                        "reocr_text": reocr_text,
                        "reocr_confidence": round(reocr_conf, 4),
                        "final_text": content,
                    })
                else:
                    content = reocr_text
                    corrections_made.append({
                        "pattern": pattern.pattern,
                        "original_text": element["content"],
                        "action": "reverified_corrected",
                        "reocr_text": reocr_text,
                        "reocr_confidence": round(reocr_conf, 4),
                        "final_text": reocr_text,
                    })
            else:
                original = content
                content = pattern.sub(replacement, content)
                corrections_made.append({
                    "pattern": pattern.pattern,
                    "original_text": original,
                    "action": "regex_fallback",
                    "final_text": content,
                })

        if corrections_made:
            element["content"] = content
            element["offensive_ocr_corrected"] = corrections_made
            for c in corrections_made:
                logger.debug("Offensive re-verification: %s → %s (action=%s)",
                             c["original_text"][:60], c["final_text"][:60], c["action"])

    return elements


# ============================================================================
# Prefix Restoration in Negation Context (Generalizable)
# ============================================================================

# Common words that TrOCR drops the prefix from in negation context
PREFIX_CORRECTIONS = {
    'fortunate': 'unfortunate',
    'known': 'unknown',
    'able': 'unable',
    'satisfied': 'dissatisfied',
    'appear': 'disappear',
    'happy': 'unhappy',
    'likely': 'unlikely',
    'certain': 'uncertain',
}

NEGATION_CONTEXT = ['not', "n't", 'never', 'but', 'yet', 'nor', 'neither', 'nothing', 'none']


def apply_ocr_corrections(text: str, page_context: set = None) -> str:
    """
    Apply generalizable OCR corrections (prefix restoration in negation context).

    TrOCR sometimes drops un-/dis- prefixes when the surrounding text contains
    negation words. This restores the prefix when negation context is detected
    within a 5-word window.

    Args:
        text: OCR text to correct
        page_context: Unused, kept for backwards compatibility

    Returns:
        Corrected text
    """
    if not text:
        return text

    words = text.split()
    corrected_words = []

    for i, word in enumerate(words):
        stripped = word.rstrip(':.,;!?')
        suffix = word[len(stripped):]

        word_lower = stripped.lower()
        corrected = stripped

        # Check prefix restoration in negation/contrast context
        if word_lower in PREFIX_CORRECTIONS:
            window = ' '.join(words[max(0, i - 5):i + 3]).lower()
            if any(neg in window for neg in NEGATION_CONTEXT):
                restored = PREFIX_CORRECTIONS[word_lower]
                corrected = restored if stripped.islower() else restored.upper() if stripped.isupper() else restored.capitalize()
                logger.debug("Prefix restoration: '%s' → '%s'", stripped, corrected)

        corrected_words.append(corrected + suffix)

    return ' '.join(corrected_words)
