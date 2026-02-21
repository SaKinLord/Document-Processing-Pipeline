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
    repair_decimal_dash_confusion,
    repair_dropped_parentheses,
    remove_consecutive_duplicate_words,
)
from .hallucination import process_hallucinations, filter_rotated_margin_text
from .table_validation import (
    filter_empty_regions,
    filter_invalid_tables,
    build_table_cells,
)
from .ocr_corrections import correct_nonword_ocr_errors
from .phone_date import normalize_phone_numbers
from .signatures import (
    replace_signature_text,
    filter_signature_overlap_garbage,
)
from src.config import CONFIG

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


def postprocess_output(output_data: Dict[str, Any], page_images=None) -> Dict[str, Any]:
    """
    Apply all post-processing to OCR output (17 stages).

    Stages:
     1. Filter empty table/layout regions
     2. Normalize underscore fill-in fields
     3. Validate table structure (score 0-100; remove < 50)
     4. Deduplicate layout regions
     5. Score and handle hallucinations (7-signal scoring)
     6. Filter rotated margin text (Bates numbers, vertical IDs)
     7. Clean text content (whitespace, Unicode, markup)
     8. Classification override (TrOCR majority → handwritten)
     9. Strip spurious TrOCR trailing periods (typed/mixed only)
    10. Repair decimal-dash confusion (.88 → -88)
    11. Repair dropped parentheses
    12. Non-word OCR correction (spell checker + confusion matrix)
    13. Replace signature text
    14. Filter signature overlap garbage
    15. Remove consecutive duplicate words (TrOCR beam search fix)
    16. Extract table cell content (assign text to structure grid)
    17. Normalize phone numbers

    Args:
        output_data: Raw OCR output dictionary
        page_images: List of PIL Images per page (optional, used for page dimension derivation)

    Returns:
        Cleaned output dictionary
    """
    for page_idx, page in enumerate(output_data.get("pages", [])):
        elements = page.get("elements", [])
        logger.info("Postprocessing page %d (%d elements)", page_idx + 1, len(elements))

        # Step 0: Sanitize element fields (input validation)
        elements = sanitize_elements(elements)

        # Extract page image once (used for page dimension derivation)
        page_image = page_images[page_idx] if page_images and page_idx < len(page_images) else None
        page_dims: Optional[Tuple[int, int]] = None
        if page_image is not None:
            page_dims = (page_image.width, page_image.height)

        # Step 1: Filter empty table/layout regions
        try:
            count_before = len(elements)
            elements = filter_empty_regions(elements)
            _log_step("filter_empty", count_before, len(elements))
        except Exception:
            logger.exception("  [filter_empty] failed — skipping step")

        # Step 2: Normalize underscore fill-in fields
        try:
            count_before = len(elements)
            elements = normalize_underscore_fields(elements)
            _log_step("normalize_underscores", count_before, len(elements))
        except Exception:
            logger.exception("  [normalize_underscores] failed — skipping step")

        # Step 3: Validate table structure (remove false positives)
        try:
            count_before = len(elements)
            elements = filter_invalid_tables(elements)
            _log_step("filter_tables", count_before, len(elements))
        except Exception:
            logger.exception("  [filter_tables] failed — skipping step")

        # Step 4: Deduplicate layout regions
        try:
            count_before = len(elements)
            elements = deduplicate_layout_regions(elements)
            _log_step("dedup_layout", count_before, len(elements))
        except Exception:
            logger.exception("  [dedup_layout] failed — skipping step")

        # Step 5: Score and handle hallucinations
        try:
            count_before = len(elements)
            elements = process_hallucinations(elements, page_dimensions=page_dims)
            _log_step("hallucinations", count_before, len(elements))
        except Exception:
            logger.exception("  [hallucinations] failed — skipping step")

        # Step 6: Filter rotated margin text (Bates numbers, vertical IDs)
        try:
            count_before = len(elements)
            elements = filter_rotated_margin_text(elements, page_dimensions=page_dims)
            _log_step("rotated_margin", count_before, len(elements))
        except Exception:
            logger.exception("  [rotated_margin] failed — skipping step")

        # Step 7: Clean text content
        try:
            count_before = len(elements)
            elements = clean_text_content(elements)
            _log_step("clean_text", count_before, len(elements))
        except Exception:
            logger.exception("  [clean_text] failed — skipping step")

        # Step 8: Post-OCR classification refinement
        # If majority of text elements were sourced from TrOCR, override to handwritten
        try:
            doc_type = page.get("document_type", "typed")
            text_els = [e for e in elements if e.get("type") == "text" and e.get("content", "").strip()]
            if text_els:
                trocr_count = sum(1 for e in text_els if e.get("source_model") == "trocr")
                trocr_ratio = trocr_count / len(text_els)
                if trocr_ratio >= CONFIG.trocr_majority_threshold and doc_type != "handwritten":
                    original_type = doc_type
                    doc_type = "handwritten"
                    page["document_type"] = "handwritten"
                    page["classification_override"] = {
                        "original": original_type,
                        "reason": "trocr_majority",
                        "trocr_ratio": round(trocr_ratio, 2)
                    }
                    logger.info("  [classification_override] %s → handwritten (TrOCR ratio: %.0f%%)",
                                original_type, trocr_ratio * 100)
        except Exception:
            logger.exception("  [classification_override] failed — skipping step")
            doc_type = page.get("document_type", "typed")

        # Step 9: Strip spurious TrOCR trailing periods (typed/mixed docs only)
        try:
            count_before = len(elements)
            elements = strip_trocr_trailing_periods(elements, document_type=doc_type)
            _log_step("trocr_period_strip", count_before, len(elements))
        except Exception:
            logger.exception("  [trocr_period_strip] failed — skipping step")

        # Step 10: Repair decimal-dash confusion (.88 → -88)
        try:
            elements = repair_decimal_dash_confusion(elements)
        except Exception:
            logger.exception("  [decimal_dash_repair] failed — skipping step")

        # Step 11: Repair dropped parentheses
        try:
            for element in elements:
                if element.get('type') == 'text' and element.get('content'):
                    element['content'] = repair_dropped_parentheses(element['content'])
            logger.debug("  [paren_repair] applied to text elements")
        except Exception:
            logger.exception("  [paren_repair] failed — skipping step")

        # Step 12: Non-word OCR correction (spell checker + confusion matrix)
        try:
            page_context = set()
            for element in elements:
                if element.get('type') == 'text' and element.get('content'):
                    for w in element['content'].split():
                        page_context.add(w.rstrip(':.,;!?').lower())
            for element in elements:
                if element.get('type') == 'text' and element.get('content'):
                    element['content'] = correct_nonword_ocr_errors(
                        element['content'], page_context=page_context
                    )
            logger.debug("  [nonword_ocr_corrections] spell-checker pass complete")
        except Exception:
            logger.exception("  [nonword_ocr] failed — skipping step")

        # Step 13: Replace signature text readings with '(signature)'
        try:
            count_before = len(elements)
            elements = replace_signature_text(elements)
            _log_step("signatures", count_before, len(elements))
        except Exception:
            logger.exception("  [signatures] failed — skipping step")

        # Step 14: Remove short garbage text overlapping signature regions
        try:
            count_before = len(elements)
            elements = filter_signature_overlap_garbage(elements)
            _log_step("sig_garbage_filter", count_before, len(elements))
        except Exception:
            logger.exception("  [sig_garbage_filter] failed — skipping step")

        # Step 15: Remove consecutive duplicate words (TrOCR beam search artifact)
        try:
            count_before = len(elements)
            elements = remove_consecutive_duplicate_words(elements)
            _log_step("dedup_words", count_before, len(elements))
        except Exception:
            logger.exception("  [dedup_words] failed — skipping step")

        # Step 16: Extract table cell content from structure data
        try:
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
        except Exception:
            logger.exception("  [table_cells] failed — skipping step")

        # Step 17: Normalize phone numbers
        try:
            count_before = len(elements)
            elements = normalize_phone_numbers(elements)
            _log_step("phones", count_before, len(elements))
        except Exception:
            logger.exception("  [phones] failed — skipping step")

        page["elements"] = elements

    return output_data
