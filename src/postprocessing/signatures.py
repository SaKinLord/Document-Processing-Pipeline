"""
Signature detection, text replacement, and garbage filtering for OCR output.

Handles replacing OCR-read signature text with '(signature)' markers,
filtering garbage fragments overlapping signature regions, and detecting
typed document indicators.
"""

import re
import logging
from typing import Dict, List

from src.utils.bbox import bbox_overlap_ratio_of_smaller

logger = logging.getLogger(__name__)


# ============================================================================
# Document Type Indicators (Text-based classification helpers)
# ============================================================================

FAX_HEADER_PATTERNS = [
    r'\b(FAX|FACSIMILE|TELECOPY)\b',
    r'\bFAX\s*(NO|NUMBER|#)?[:\s]*\(?\d{3}\)?[-\s]?\d{3}[-\s]?\d{4}',
    r'\b(FROM|TO|DATE|RE|SUBJECT|PAGES?)\s*:',
]

TYPED_INDICATOR_PATTERNS = [
    r'\bDEPARTMENT\s+OF\b',
    r'\b(FORM|APPLICATION|CERTIFICATE)\s+\d+',
    r'\bOFFICIAL\s+USE\s+ONLY\b',
    r'\bPRINT\s+NAME\b',
    r'\bSIGNATURE\s+DATE\b',
    r'\b(?:PAGE|PG|P)\s*\d+\s*(?:OF|/)\s*\d+',
]


def detect_typed_document_indicators(text: str) -> dict:
    """
    Detect text patterns that indicate a typed/form/fax document.

    This is used to catch misclassified typed documents that were
    incorrectly classified as handwritten based on image features.
    """
    result = {
        'is_likely_typed': False,
        'fax_indicators': [],
        'form_indicators': [],
        'confidence_boost': 0.0
    }

    if not text:
        return result

    text_upper = text.upper()

    for pattern in FAX_HEADER_PATTERNS:
        if re.search(pattern, text_upper):
            result['fax_indicators'].append(pattern)

    for pattern in TYPED_INDICATOR_PATTERNS:
        if re.search(pattern, text_upper):
            result['form_indicators'].append(pattern)

    fax_count = len(result['fax_indicators'])
    form_count = len(result['form_indicators'])

    if fax_count >= 2:
        result['confidence_boost'] = 0.25
        result['is_likely_typed'] = True
    elif fax_count == 1 and form_count >= 1:
        result['confidence_boost'] = 0.20
        result['is_likely_typed'] = True
    elif form_count >= 3:
        result['confidence_boost'] = 0.15
        result['is_likely_typed'] = True
    elif fax_count == 1 or form_count >= 2:
        result['confidence_boost'] = 0.10

    return result


# ============================================================================
# Signature Text Replacement
# ============================================================================

SIGNATURE_LABEL_RE = re.compile(
    r'((?:RECEIVED|SIGNED)\s+BY\s*:'
    r'|SIGNATURE\s+OF\s+[\w\s]*?(?:CONSIGNEE|INITIATOR|APPLICANT|AUTHORIZED(?:\s+\w+)?)\s*:'
    r'|SIGNATURE\s*:)\s*'
    r'([A-Za-z]+(?:\s+[A-Za-z]+){0,2})'
    r'(?=\s+DATE\s*:|$)',
    re.IGNORECASE
)


def replace_signature_text(elements: List[Dict]) -> List[Dict]:
    """
    Replace OCR-read signature text with '(signature)'.

    Matches patterns like 'RECEIVED BY: Some Name DATE:' and replaces
    the name portion with '(signature)'.
    """
    for element in elements:
        if element.get('type') == 'text' and element.get('content'):
            element['content'] = SIGNATURE_LABEL_RE.sub(
                lambda m: m.group(1) + ' (signature)', element['content']
            )
    return elements


# ============================================================================
# Signature Overlap Garbage Filter
# ============================================================================

_DATE_LIKE_RE = re.compile(
    r'^\d{1,2}[-/][A-Za-z]{3,9}[-/]\d{2,4}$'
    r'|^\d{1,2}[-/]\d{1,2}[-/]\d{2,4}$'
)


def filter_signature_overlap_garbage(elements: List[Dict]) -> List[Dict]:
    """
    Remove single-word garbage fragments that overlap with signature elements.

    When Surya reads cursive signatures, it often produces short garbage strings
    like 'elevens.', 'not', '100' that are too "normal" for the hallucination
    scorer but clearly wrong when correlated with Florence-2's signature detection.

    Only uses 'signature' visual elements (not logo/seal).
    """
    signature_bboxes = []
    for el in elements:
        if el.get('type', '').lower() == 'signature':
            bbox = el.get('bbox')
            if bbox and len(bbox) == 4:
                signature_bboxes.append(bbox)

    if not signature_bboxes:
        return elements

    filtered = []
    for el in elements:
        if el.get('type') != 'text':
            filtered.append(el)
            continue

        content = el.get('content', '').strip()
        word_count = len(content.split()) if content else 0

        if word_count != 1:
            filtered.append(el)
            continue

        if _DATE_LIKE_RE.match(content.rstrip('.,')):
            filtered.append(el)
            continue

        el_bbox = el.get('bbox')
        if not el_bbox or len(el_bbox) != 4:
            filtered.append(el)
            continue

        overlaps_sig = False
        for sig_bbox in signature_bboxes:
            overlap = bbox_overlap_ratio_of_smaller(el_bbox, sig_bbox)
            if overlap > 0.50:
                overlaps_sig = True
                break

        if overlaps_sig:
            logger.debug("Removed '%s' (overlaps signature region)", content)
        else:
            filtered.append(el)

    return filtered
