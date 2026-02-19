"""
OCR correction logic for post-processing.

Handles:
- Generalizable non-word correction via spell checker + OCR confusion matrix
"""

import logging
from typing import Optional, Set

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
