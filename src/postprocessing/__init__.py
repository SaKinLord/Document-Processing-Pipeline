"""
Post-processing package for OCR output.

Re-exports all public functions for backwards-compatible imports.
"""

from .pipeline import postprocess_output, deduplicate_layout_regions
from .normalization import (
    normalize_underscores,
    normalize_underscore_fields,
    normalize_punctuation_spacing,
    clean_text_content,
    fix_encoding_issues,
    strip_trocr_trailing_periods,
    repair_dropped_parentheses,
    remove_consecutive_duplicate_words,
)
from .hallucination import (
    process_hallucinations,
    filter_rotated_margin_text,
    calculate_hallucination_score,
    check_character_patterns,
    is_valid_text,
    has_repetition_pattern,
)
from .table_validation import (
    filter_empty_regions,
    filter_invalid_tables,
    validate_table_structure,
    calculate_text_density,
    cluster_positions,
    detect_column_alignment,
    detect_row_alignment,
    check_grid_coverage,
    calculate_structure_score,
)
from .ocr_corrections import correct_nonword_ocr_errors
from .phone_date import (
    validate_phone_number,
    add_phone_validation_to_element,
    validate_date_format,
    add_date_validation_to_element,
    normalize_phone_numbers,
    extract_phone_numbers,
    is_date_or_zip,
    detect_phone_type,
)
from .signatures import (
    replace_signature_text,
    filter_signature_overlap_garbage,
)
__all__ = [
    # Pipeline
    'postprocess_output',
    'deduplicate_layout_regions',
    # Normalization
    'normalize_underscores',
    'normalize_underscore_fields',
    'normalize_punctuation_spacing',
    'clean_text_content',
    'fix_encoding_issues',
    'strip_trocr_trailing_periods',
    'repair_dropped_parentheses',
    'remove_consecutive_duplicate_words',
    # Hallucination
    'process_hallucinations',
    'filter_rotated_margin_text',
    'calculate_hallucination_score',
    'check_character_patterns',
    'is_valid_text',
    'has_repetition_pattern',
    # Table validation
    'filter_empty_regions',
    'filter_invalid_tables',
    'validate_table_structure',
    'calculate_text_density',
    'cluster_positions',
    'detect_column_alignment',
    'detect_row_alignment',
    'check_grid_coverage',
    'calculate_structure_score',
    # OCR corrections
    'correct_nonword_ocr_errors',
    # Phone/Date
    'validate_phone_number',
    'add_phone_validation_to_element',
    'validate_date_format',
    'add_date_validation_to_element',
    'normalize_phone_numbers',
    'extract_phone_numbers',
    'is_date_or_zip',
    'detect_phone_type',
    # Signatures
    'replace_signature_text',
    'filter_signature_overlap_garbage',
]
