"""
Post-processing pipeline orchestrator.

Coordinates all post-processing steps in the correct order.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple

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
    build_table_cells,
)
from .ocr_corrections import (
    filter_offensive_ocr_misreads,
    apply_ocr_corrections,
    apply_multi_word_ocr_corrections,
)
from .phone_date import normalize_phone_numbers
from .signatures import (
    replace_signature_text,
    filter_signature_overlap_garbage,
    detect_typed_document_indicators,
)
logger = logging.getLogger(__name__)


def _log_step(name: str, before: int, after: int) -> None:
    """Log a postprocessing step, using INFO when elements are removed."""
    delta = before - after
    if delta > 0:
        logger.info("  [%s] %d → %d elements (removed %d)", name, before, after, delta)
    else:
        logger.debug("  [%s] %d elements (no change)", name, before)


def sanitize_elements(elements: List[Dict]) -> List[Dict]:
    """Validate and coerce element fields at the pipeline boundary.

    Ensures every element has the expected field types so downstream steps
    don't need to guard against malformed input:
    - ``type``: must be a string (default ``"unknown"``)
    - ``content``: must be a string if present (coerced with ``str()``)
    - ``bbox``: must be a list of 4 floats (set to ``[]`` if invalid)
    - ``confidence``: must be a float in [0, 1] (clamped)
    """
    for element in elements:
        # type
        if not isinstance(element.get("type"), str):
            element["type"] = "unknown"

        # content
        if "content" in element and not isinstance(element["content"], str):
            element["content"] = str(element["content"])

        # bbox
        bbox = element.get("bbox")
        if isinstance(bbox, (list, tuple)):
            try:
                if len(bbox) >= 4:
                    element["bbox"] = [float(bbox[0]), float(bbox[1]),
                                       float(bbox[2]), float(bbox[3])]
                else:
                    element["bbox"] = []
            except (TypeError, ValueError):
                element["bbox"] = []
        elif bbox is not None:
            element["bbox"] = []

        # confidence
        conf = element.get("confidence")
        if conf is not None:
            try:
                conf = float(conf)
                element["confidence"] = max(0.0, min(1.0, conf))
            except (TypeError, ValueError):
                element["confidence"] = 0.0

    return elements


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
    3.55. Extract table cell content (assign text to structure grid)
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
        logger.info("Postprocessing page %d (%d elements)", page_idx + 1, len(elements))

        # Step -1: Sanitize element fields (validate at the boundary)
        elements = sanitize_elements(elements)

        # Extract page image once (used by offensive filter + dimension derivation)
        page_image = page_images[page_idx] if page_images and page_idx < len(page_images) else None
        page_dims: Optional[Tuple[int, int]] = None
        if page_image is not None:
            page_dims = (page_image.width, page_image.height)

        # Step 0: Filter empty table/layout regions
        count_before = len(elements)
        elements = filter_empty_regions(elements)
        _log_step("filter_empty", count_before, len(elements))

        # Step 0.25: Normalize underscore fill-in fields
        count_before = len(elements)
        elements = normalize_underscore_fields(elements)
        _log_step("normalize_underscores", count_before, len(elements))

        # Step 0.5: Validate table structure (remove false positives)
        count_before = len(elements)
        elements = filter_invalid_tables(elements)
        _log_step("filter_tables", count_before, len(elements))

        # Step 1: Deduplicate layout regions
        count_before = len(elements)
        elements = deduplicate_layout_regions(elements)
        _log_step("dedup_layout", count_before, len(elements))

        # Step 2: Score and handle hallucinations
        count_before = len(elements)
        elements = process_hallucinations(elements, page_dimensions=page_dims)
        _log_step("hallucinations", count_before, len(elements))

        # Step 2.1: Filter rotated margin text (Bates numbers, vertical IDs)
        count_before = len(elements)
        elements = filter_rotated_margin_text(elements, page_dimensions=page_dims)
        _log_step("rotated_margin", count_before, len(elements))

        # Step 2.5: Filter offensive OCR misreads (with source image re-verification)
        count_before = len(elements)
        elements = filter_offensive_ocr_misreads(elements, page_image=page_image,
                                                  handwriting_recognizer=handwriting_recognizer)
        _log_step("offensive_filter", count_before, len(elements))

        # Step 3: Clean text content
        count_before = len(elements)
        elements = clean_text_content(elements)
        _log_step("clean_text", count_before, len(elements))

        # Step 3.05: Strip spurious TrOCR trailing periods (typed/mixed docs only)
        doc_type = page.get("document_type", "typed")
        count_before = len(elements)
        elements = strip_trocr_trailing_periods(elements, document_type=doc_type)
        _log_step("trocr_period_strip", count_before, len(elements))

        # Step 3.1: Repair dropped parentheses
        for element in elements:
            if element.get('type') == 'text' and element.get('content'):
                element['content'] = repair_dropped_parentheses(element['content'])
        logger.debug("  [paren_repair] applied to text elements")

        # Step 3.25: Apply OCR corrections (with slash-compound splitting)
        page_context = set()
        for element in elements:
            if element.get('type') == 'text' and element.get('content'):
                for w in element['content'].split():
                    page_context.add(w.rstrip(':.,;!?').lower())
        for element in elements:
            if element.get('type') == 'text' and element.get('content'):
                element['content'] = apply_ocr_corrections(
                    element['content'], page_context=page_context
                )
        logger.debug("  [ocr_corrections] applied to text elements")

        # Step 3.26: Apply multi-word proper noun corrections
        for element in elements:
            if element.get('type') == 'text' and element.get('content'):
                element['content'] = apply_multi_word_ocr_corrections(element['content'], page_context=page_context)
        logger.debug("  [multi_word_corrections] applied to text elements")

        # Step 3.3: Replace signature text readings with '(signature)'
        count_before = len(elements)
        elements = replace_signature_text(elements)
        _log_step("signatures", count_before, len(elements))

        # Step 3.35: Remove short garbage text overlapping signature regions
        count_before = len(elements)
        elements = filter_signature_overlap_garbage(elements)
        _log_step("sig_garbage_filter", count_before, len(elements))

        # Step 3.5: Remove consecutive duplicate words (TrOCR beam search artifact)
        count_before = len(elements)
        elements = remove_consecutive_duplicate_words(elements)
        _log_step("dedup_words", count_before, len(elements))

        # Step 3.55: Extract table cell content from structure data
        text_elements = [e for e in elements if e.get("type") == "text"]
        cell_count = 0
        for element in elements:
            if element.get("type") == "table" and "structure" in element:
                cells = build_table_cells(element, text_elements)
                if cells:
                    element["cells"] = cells
                    element["num_rows"] = max(c["row"] for c in cells) + 1
                    element["num_columns"] = max(c["col"] for c in cells) + 1
                    cell_count += len(cells)
                del element["structure"]
        if cell_count:
            logger.info("  [table_cells] extracted %d cells", cell_count)
        else:
            logger.debug("  [table_cells] no tables with structure data")

        # Step 4: Normalize phone numbers
        count_before = len(elements)
        elements = normalize_phone_numbers(elements)
        _log_step("phones", count_before, len(elements))

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
