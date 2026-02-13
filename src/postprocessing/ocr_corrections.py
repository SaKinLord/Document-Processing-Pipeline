"""
OCR correction logic for handwritten and typed documents.

Handles single-word confusion corrections, multi-word proper noun corrections,
prefix restoration, offensive OCR misread filtering with cross-model re-verification,
and slash-compound word splitting.
"""

import re
import logging
from typing import Dict, List

from .normalization import _clean_trocr_trailing_period

logger = logging.getLogger(__name__)

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
                crop = page_image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
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

    return elements


# ============================================================================
# Handwriting OCR Correction
# ============================================================================

# Common TrOCR confusion pairs: (wrong_word, correct_word, context_words)
OCR_CONFUSION_CORRECTIONS = [
    ('last', 'lost', ['property', 'found', 'missing', 'retrieve', 'retrieving']),
    ('book', 'look', ['neat', 'have', 'take', 'good', 'had', 'that']),
    ('intact', 'in fact', ['no', 'not', 'but', 'actually']),
    ('form', 'from', ['received', 'sent', 'get', 'letter', 'came']),
    ('bock', 'back', ['come', 'go', 'went', 'came', 'get']),
    ('sane', 'same', ['the', 'at', 'time', 'way']),
    ('dime', 'time', ['the', 'at', 'same', 'every', 'any']),
    ('gambetta', 'lambretta', ['negresco', 'beach', 'parked', 'opposite', 'promenade']),
    ('negress', 'negresco', ['lambretta', 'beach', 'promenade', 'opposite', 'swim']),
    ('angles', 'anglais', ['promenade', 'des', 'nice', 'walls', 'spumed']),
    ('patriot', 'patriote', ['read', 'nice', 'beach', 'catastrophe', 'about']),
    ('lobito', 'losito', ['cc', 'tahmaseb', 'baroody', 'stevens', 'registration']),
    ('proliffements', 'requirements', []),
    ('leaved', 'leaked', ['condition', 'shipment', 'good', 'broken', 'article']),
    ('overlap', 'overwrap', ['filter', 'pack', 'type', 'flavoring']),
    ('depariment', 'department', []),
    ('engine', 'broken', ['condition', 'shipment', 'good', 'leaked']),
    ('atterney', 'attorney', []),
    ('decamps', 'delchamps', ['stores', 'account', 'maverick', 'distribution', 'region']),
    ('antler', 'dantzler', ['stores', 'account', 'maverick', 'region', 'distribution']),
    ('indoor', 'ind/lor', ['volume', 'stores', 'account', 'maverick', 'distribution']),
    ('approve', 'approx', ['circulation', 'geographical', 'redemption', 'coupon']),
    ('incipient', 'recipient', ['intended', 'disclosure', 'confidential', 'privileged']),
    ('probable', 'prohibited', ['disclosure', 'confidential', 'telecopy', 'privileged']),
]

# Multi-word OCR corrections for proper nouns spanning multiple tokens.
MULTI_WORD_OCR_CORRECTIONS = [
    ('HAVENS GERMAN', 'HAGENS BERMAN', []),
    ('Havens German', 'Hagens Berman', []),
    ('Steve W. German', 'Steve W. Berman', []),
    ('Meyer G. Follow', 'Meyer G. Koplow', []),
    ('Rose & Kate', 'Rosen & Katz', []),
    ('Martin Harrington', 'Martin Barrington', []),
    ('Martin Warrington', 'Martin Barrington', []),
    ('Farewell', 'Wardwell', ['davis', 'polk', 'law', 'counsel', 'firm']),
    ('Ronald Einstein', 'Ronald Milstein', []),
    ('Charles A. Bit', 'Charles A. Blixt', []),
    ('Style Oil', 'Sayle Oil', []),
    ('Try Green', 'Autry Greer', ['stores', 'region', 'account', 'distribution', 'maverick']),
    ('Win Dixie', 'Winn Dixie', []),
    ('Compact Foods', 'Compac Foods', []),
]

# Words that commonly appear with prefixes that TrOCR misses
PREFIX_CORRECTIONS = {
    'cuckolded': 'uncuckolded',
    'robbed': 'unrobbed',
    'fortunate': 'unfortunate',
    'known': 'unknown',
    'able': 'unable',
    'satisfied': 'dissatisfied',
    'appear': 'disappear',
    'happy': 'unhappy',
    'likely': 'unlikely',
    'certain': 'uncertain',
}

# Extended negation context for prefix restoration
NEGATION_CONTEXT = ['not', "n't", 'never', 'but', 'yet', 'nor', 'neither', 'nothing', 'none', 'chosen']


def apply_ocr_corrections_handwritten(text: str, is_handwritten: bool = False, page_context: set = None) -> str:
    """
    Apply OCR-specific corrections for handwritten documents only.

    Uses flexible context matching -- looks for context words within a 5-word
    window around the potentially confused word. Falls back to page-level
    context when the element has too few words for local context matching.

    Args:
        text: OCR text to correct
        is_handwritten: Whether the source document is handwritten
        page_context: Optional set of lowercased words from all text elements
                      on the page, used as fallback for short/standalone elements

    Returns:
        Corrected text (unchanged if not handwritten)
    """
    if not is_handwritten or not text:
        return text

    words = text.split()
    corrected_words = []
    context_window_size = 5

    for i, word in enumerate(words):
        stripped = word.rstrip(':.,;!?')
        suffix = word[len(stripped):]

        window_start = max(0, i - context_window_size)
        window_end = min(len(words), i + context_window_size + 1)
        context_words = set(w.rstrip(':.,;!?').lower() for w in words[window_start:window_end])

        # Handle compound words joined by '/' or '-'
        for sep in ('/', '-'):
            if sep in stripped:
                parts = stripped.split(sep)
                corrected_parts = []
                for part in parts:
                    part_lower = part.lower()
                    part_corrected = part
                    for wrong, correct, context in OCR_CONFUSION_CORRECTIONS:
                        if part_lower == wrong:
                            if not context or any(ctx in context_words for ctx in context) or (page_context and any(ctx in page_context for ctx in context)):
                                part_corrected = correct if part.islower() else correct.upper() if part.isupper() else correct.capitalize()
                                break
                    corrected_parts.append(part_corrected)
                corrected_words.append(sep.join(corrected_parts) + suffix)
                break
        else:
            pass

        # If we handled a compound word above, skip normal processing
        if '/' in stripped or '-' in stripped:
            continue

        word_lower = stripped.lower()
        corrected = stripped
        context_words.discard(word_lower)

        for wrong, correct, context in OCR_CONFUSION_CORRECTIONS:
            if word_lower == wrong:
                if not context or any(ctx in context_words for ctx in context) or (page_context and any(ctx in page_context for ctx in context)):
                    corrected = correct if stripped.islower() else correct.upper() if stripped.isupper() else correct.capitalize()
                    break

        # Check prefix restoration in negation/contrast context
        if word_lower in PREFIX_CORRECTIONS:
            window = ' '.join(words[max(0, i - 5):i + 3]).lower()
            if any(neg in window for neg in NEGATION_CONTEXT):
                restored = PREFIX_CORRECTIONS[word_lower]
                corrected = restored if stripped.islower() else restored.upper() if stripped.isupper() else restored.capitalize()

        corrected_words.append(corrected + suffix)

    return ' '.join(corrected_words)


def apply_multi_word_ocr_corrections(text: str, page_context: set = None) -> str:
    """
    Apply multi-word OCR corrections for proper nouns spanning multiple tokens.

    Args:
        text: OCR text to correct
        page_context: Optional set of lowercased words from all text elements

    Returns:
        Text with multi-word corrections applied
    """
    if not text:
        return text

    text_lower = text.lower()

    for wrong, correct, context in MULTI_WORD_OCR_CORRECTIONS:
        pattern = re.compile(re.escape(wrong), re.IGNORECASE)
        if pattern.search(text):
            if not context or any(ctx in text_lower for ctx in context) or (page_context and any(ctx in page_context for ctx in context)):
                text = pattern.sub(correct, text)
                text_lower = text.lower()

    return text
