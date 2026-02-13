"""
Post-processing pipeline orchestrator.

Coordinates all post-processing steps in the correct order.
"""

import logging
from typing import Dict, Any

from .normalization import (
    normalize_underscore_fields,
    clean_text_content,
    strip_trocr_trailing_periods,
    repair_dropped_parentheses,
    remove_consecutive_duplicate_words,
)
from .hallucination import process_hallucinations, filter_rotated_margin_text
from .table_validation import (
    filter_empty_regions,
    filter_invalid_tables,
    promote_layout_regions_to_tables,
)
from .ocr_corrections import (
    filter_offensive_ocr_misreads,
    apply_ocr_corrections_handwritten,
    apply_multi_word_ocr_corrections,
)
from .phone_date import normalize_phone_numbers
from .signatures import (
    replace_signature_text,
    filter_signature_overlap_garbage,
    detect_typed_document_indicators,
)
from .helpers import bbox_overlap

logger = logging.getLogger(__name__)


def deduplicate_layout_regions(elements):
    """
    Remove duplicate layout_region elements with same bbox.
    Keeps one representative region per unique bbox.
    """
    seen_bboxes = set()
    deduplicated = []

    for element in elements:
        if element.get("type") == "layout_region":
            bbox = element.get("bbox", [])
            bbox_key = tuple(round(x, 2) for x in bbox) if bbox else ()

            if bbox_key in seen_bboxes:
                continue
            seen_bboxes.add(bbox_key)

        deduplicated.append(element)

    return deduplicated


def postprocess_output(output_data: Dict[str, Any], page_images=None,
                       handwriting_recognizer=None) -> Dict[str, Any]:
    """
    Apply all post-processing to OCR output.

    Steps:
    0. Filter empty table/layout regions (remove hallucinations)
    0.25. Normalize underscore fill-in fields
    0.5. Validate table structure (remove false positive tables)
    0.6. Heuristic table promotion (detect missed borderless tables)
    1. Deduplicate layout regions
    2. Score and handle hallucinations (with margin awareness)
    2.1. Filter rotated margin text (Bates numbers, vertical IDs)
    2.5. Filter offensive OCR misreads (P0 reputational risk, with re-verification)
    3. Clean text content
    3.05. Strip spurious TrOCR trailing periods (typed/mixed docs only)
    3.1. Repair dropped parentheses (P3)
    3.25. Apply handwritten OCR corrections (with slash-compound splitting)
    3.26. Apply multi-word proper noun corrections (P1)
    3.3. Replace signature text
    3.35. Filter signature overlap garbage (short fragments on cursive signatures)
    3.5. Remove consecutive duplicate words (TrOCR beam search fix)
    4. Normalize phone numbers

    Args:
        output_data: Raw OCR output dictionary
        page_images: List of PIL Images per page (optional, enables offensive filter re-verification)
        handwriting_recognizer: TrOCR recognizer instance (optional)

    Returns:
        Cleaned output dictionary
    """
    for page_idx, page in enumerate(output_data.get("pages", [])):
        elements = page.get("elements", [])

        # Step 0: Filter empty table/layout regions
        elements = filter_empty_regions(elements)

        # Step 0.25: Normalize underscore fill-in fields
        elements = normalize_underscore_fields(elements)

        # Step 0.5: Validate table structure (remove false positives)
        elements = filter_invalid_tables(elements)

        # Step 0.6: Heuristic table promotion (detect missed tables)
        elements = promote_layout_regions_to_tables(elements)

        # Step 1: Deduplicate layout regions
        elements = deduplicate_layout_regions(elements)

        # Step 2: Score and handle hallucinations
        elements = process_hallucinations(elements)

        # Step 2.1: Filter rotated margin text (Bates numbers, vertical IDs)
        elements = filter_rotated_margin_text(elements)

        # Step 2.5: Filter offensive OCR misreads (with source image re-verification)
        page_image = page_images[page_idx] if page_images and page_idx < len(page_images) else None
        elements = filter_offensive_ocr_misreads(elements, page_image=page_image,
                                                  handwriting_recognizer=handwriting_recognizer)

        # Step 3: Clean text content
        elements = clean_text_content(elements)

        # Step 3.05: Strip spurious TrOCR trailing periods (typed/mixed docs only)
        doc_type = page.get("document_type", "typed")
        elements = strip_trocr_trailing_periods(elements, document_type=doc_type)

        # Step 3.1: Repair dropped parentheses
        for element in elements:
            if element.get('type') == 'text' and element.get('content'):
                element['content'] = repair_dropped_parentheses(element['content'])

        # Step 3.25: Apply OCR corrections (with slash-compound splitting)
        page_context = set()
        for element in elements:
            if element.get('type') == 'text' and element.get('content'):
                for w in element['content'].split():
                    page_context.add(w.rstrip(':.,;!?').lower())
        for element in elements:
            if element.get('type') == 'text' and element.get('content'):
                element['content'] = apply_ocr_corrections_handwritten(
                    element['content'], is_handwritten=True, page_context=page_context
                )

        # Step 3.26: Apply multi-word proper noun corrections
        for element in elements:
            if element.get('type') == 'text' and element.get('content'):
                element['content'] = apply_multi_word_ocr_corrections(element['content'], page_context=page_context)

        # Step 3.3: Replace signature text readings with '(signature)'
        elements = replace_signature_text(elements)

        # Step 3.35: Remove short garbage text overlapping signature regions
        elements = filter_signature_overlap_garbage(elements)

        # Step 3.5: Remove consecutive duplicate words (TrOCR beam search artifact)
        elements = remove_consecutive_duplicate_words(elements)

        # Step 4: Normalize phone numbers
        elements = normalize_phone_numbers(elements)

        # Step 5: Classification refinement (detect misclassified typed docs)
        all_text = ' '.join([e.get('content', '') for e in elements if e.get('type') == 'text'])
        typed_indicators = detect_typed_document_indicators(all_text)

        if typed_indicators['is_likely_typed']:
            page['classification_refinement'] = {
                'likely_typed': True,
                'fax_indicators': len(typed_indicators['fax_indicators']),
                'form_indicators': len(typed_indicators['form_indicators']),
                'suggested_boost': typed_indicators['confidence_boost']
            }

        page["elements"] = elements

    return output_data
