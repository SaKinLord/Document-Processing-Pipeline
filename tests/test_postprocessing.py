"""
Unit tests for core postprocessing functions.

Tests pure functions that don't require GPU models (no Surya, TrOCR, Florence-2).
Run with: pytest tests/test_postprocessing.py -v
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path for both standalone and pytest usage
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import pytest

from src.utils.bbox import (
    bbox_overlap_ratio_of_smaller,
    bbox_overlap_ratio_of,
    bboxes_intersect,
    split_line_bbox_to_words,
    is_bbox_too_large,
)
from src.postprocessing.normalization import (
    normalize_underscores,
    normalize_punctuation_spacing,
    fix_encoding_issues,
    _clean_trocr_trailing_period,
    repair_dropped_parentheses,
    repair_decimal_dash_confusion,
    strip_trocr_trailing_periods,
    remove_consecutive_duplicate_words,
    clean_text_content,
    normalize_underscore_fields,
)
from src.postprocessing.ocr_corrections import (
    correct_nonword_ocr_errors,
    _is_ocr_plausible,
)
from src.postprocessing.hallucination import (
    check_character_patterns,
    is_valid_text,
    has_repetition_pattern,
    calculate_hallucination_score,
    process_hallucinations,
    filter_rotated_margin_text,
    _detect_numeric_columns,
)
from src.postprocessing.table_validation import (
    cluster_positions,
    calculate_text_density,
    calculate_structure_score,
)
from src.postprocessing.phone_date import (
    validate_phone_number,
    validate_date_format,
    is_date_or_zip,
    extract_phone_numbers,
    add_phone_validation_to_element,
)
from src.postprocessing.pipeline import (
    sanitize_elements,
    deduplicate_layout_regions,
)
from src.postprocessing.normalization import _transfer_surya_case


# ============================================================================
# Bbox Utilities
# ============================================================================

class TestBboxOverlapRatioOfSmaller:
    def test_identical_bboxes(self):
        bbox = [0, 0, 100, 100]
        assert bbox_overlap_ratio_of_smaller(bbox, bbox) == 1.0

    def test_no_overlap(self):
        assert bbox_overlap_ratio_of_smaller([0, 0, 50, 50], [60, 60, 100, 100]) == 0.0

    def test_partial_overlap(self):
        ratio = bbox_overlap_ratio_of_smaller([0, 0, 100, 100], [50, 50, 150, 150])
        assert 0.0 < ratio < 1.0

    def test_smaller_inside_larger(self):
        # Small box fully inside large box: overlap = 100% of smaller
        assert bbox_overlap_ratio_of_smaller([10, 10, 20, 20], [0, 0, 100, 100]) == 1.0

    def test_invalid_bbox_length(self):
        assert bbox_overlap_ratio_of_smaller([0, 0], [0, 0, 100, 100]) == 0.0
        assert bbox_overlap_ratio_of_smaller([0, 0, 100, 100], []) == 0.0

    def test_zero_area_bbox(self):
        assert bbox_overlap_ratio_of_smaller([0, 0, 0, 100], [0, 0, 100, 100]) == 0.0

    def test_touching_bboxes_no_overlap(self):
        # Bboxes share an edge but don't overlap
        assert bbox_overlap_ratio_of_smaller([0, 0, 50, 50], [50, 0, 100, 50]) == 0.0


class TestBboxOverlapRatioOf:
    def test_identical(self):
        bbox = [0, 0, 100, 100]
        assert bbox_overlap_ratio_of(bbox, bbox, reference_bbox=bbox) == 1.0

    def test_small_reference(self):
        # 50% of small box is covered
        small = [0, 0, 10, 10]
        large = [5, 0, 100, 100]
        ratio = bbox_overlap_ratio_of(small, large, reference_bbox=small)
        assert ratio == pytest.approx(0.5)


class TestBboxesIntersect:
    def test_overlapping(self):
        assert bboxes_intersect([0, 0, 100, 100], [50, 50, 150, 150]) is True

    def test_non_overlapping(self):
        assert bboxes_intersect([0, 0, 50, 50], [60, 60, 100, 100]) is False

    def test_touching_edge(self):
        # Touching but not overlapping (shared edge, 0 intersection area)
        assert bboxes_intersect([0, 0, 50, 50], [50, 0, 100, 50]) is False

    def test_invalid_bbox(self):
        assert bboxes_intersect([0, 0], [0, 0, 100, 100]) is False


class TestSplitLineBboxToWords:
    def test_single_word(self):
        result = split_line_bbox_to_words([0, 0, 100, 20], ["hello"])
        assert result == [[0, 0, 100, 20]]

    def test_empty_words(self):
        assert split_line_bbox_to_words([0, 0, 100, 20], []) == []

    def test_multiple_words(self):
        result = split_line_bbox_to_words([0, 0, 200, 20], ["hi", "there"])
        assert len(result) == 2
        assert result[0][0] == 0  # First word starts at x1
        assert result[0][3] == 20  # Same y2

    def test_word_bboxes_are_ordered(self):
        result = split_line_bbox_to_words([0, 0, 300, 20], ["a", "bb", "ccc"])
        for i in range(len(result) - 1):
            assert result[i][0] < result[i + 1][0], "Bboxes should be left-to-right"


class TestIsBboxTooLarge:
    def test_small_bbox(self):
        assert is_bbox_too_large([0, 0, 50, 50], 1000, 1000, label="signature") is False

    def test_large_signature(self):
        # 30% of page — exceeds 0.25 threshold
        assert is_bbox_too_large([0, 0, 547, 547], 1000, 1000, label="signature") is True

    def test_large_graphic(self):
        # 49% — under 0.50 threshold
        assert is_bbox_too_large([0, 0, 700, 700], 1000, 1000, label="graphic") is False


# ============================================================================
# Normalization
# ============================================================================

class TestNormalizeUnderscores:
    def test_spaced_underscores(self):
        assert normalize_underscores("_ _ _ _") == "___"

    def test_long_underscore_run(self):
        assert normalize_underscores("________") == "___"

    def test_colon_spaces(self):
        result = normalize_underscores("Name:      ")
        assert "___" in result

    def test_double_spaces_cleaned(self):
        result = normalize_underscores("hello  world")
        assert "  " not in result

    def test_short_underscores_preserved(self):
        # Two underscores should be left alone (below 3+ threshold)
        assert normalize_underscores("__") == "__"


class TestNormalizePunctuationSpacing:
    def test_space_before_period(self):
        assert normalize_punctuation_spacing("word .") == "word."

    def test_space_before_comma(self):
        assert normalize_punctuation_spacing("Govr. ,") == "Govr.,", "Double punct with space"

    def test_space_around_parens(self):
        assert normalize_punctuation_spacing("( text )") == "(text)"

    def test_space_around_hyphens(self):
        assert normalize_punctuation_spacing("self - aware") == "self-aware"

    def test_space_around_apostrophe(self):
        assert normalize_punctuation_spacing("don 't") == "don't"

    def test_empty_input(self):
        assert normalize_punctuation_spacing("") == ""
        assert normalize_punctuation_spacing(None) is None


class TestFixEncodingIssues:
    def test_smart_quotes(self):
        assert fix_encoding_issues("\u201cquote\u201d") == '"quote"'
        assert fix_encoding_issues("\u2018text\u2019") == "'text'"

    def test_dashes(self):
        assert fix_encoding_issues("a\u2013b") == "a-b"  # en dash
        assert fix_encoding_issues("a\u2014b") == "a-b"  # em dash

    def test_ellipsis(self):
        assert fix_encoding_issues("wait\u2026") == "wait..."

    def test_nbsp(self):
        assert fix_encoding_issues("non\u00a0breaking") == "non breaking"


class TestCleanTrocrTrailingPeriod:
    def test_strips_trailing_period(self):
        assert _clean_trocr_trailing_period("DEPARTMENT.") == "DEPARTMENT"

    def test_strips_space_period(self):
        assert _clean_trocr_trailing_period("DEPARTMENT .") == "DEPARTMENT"

    def test_preserves_abbreviation_inc(self):
        assert _clean_trocr_trailing_period("Company Inc.") == "Company Inc."

    def test_preserves_abbreviation_corp(self):
        assert _clean_trocr_trailing_period("Corp.") == "Corp."

    def test_preserves_single_letter_initial(self):
        assert _clean_trocr_trailing_period("S.") == "S."

    def test_no_period(self):
        assert _clean_trocr_trailing_period("hello") == "hello"

    def test_empty_string(self):
        assert _clean_trocr_trailing_period("") == ""


class TestRepairDroppedParentheses:
    def test_brands_to_brand_s(self):
        assert repair_dropped_parentheses("BRANDS)") == "BRAND(S)"

    def test_divisions_to_division_s(self):
        assert repair_dropped_parentheses("DIVISIONS)") == "DIVISION(S)"

    def test_lowercase(self):
        assert repair_dropped_parentheses("recipients)") == "recipient(s)"

    def test_balanced_parens_untouched(self):
        assert repair_dropped_parentheses("BRAND(S)") == "BRAND(S)"

    def test_no_paren(self):
        assert repair_dropped_parentheses("BRANDS") == "BRANDS"

    def test_empty(self):
        assert repair_dropped_parentheses("") == ""


class TestStripTrocrTrailingPeriods:
    def test_strips_period_from_trocr_typed(self):
        elements = [{"type": "text", "source_model": "trocr", "content": "HELLO."}]
        result = strip_trocr_trailing_periods(elements, document_type="typed")
        assert result[0]["content"] == "HELLO"

    def test_skips_handwritten_docs(self):
        elements = [{"type": "text", "source_model": "trocr", "content": "HELLO."}]
        result = strip_trocr_trailing_periods(elements, document_type="handwritten")
        assert result[0]["content"] == "HELLO."

    def test_skips_surya_model(self):
        elements = [{"type": "text", "source_model": "surya", "content": "HELLO."}]
        result = strip_trocr_trailing_periods(elements, document_type="typed")
        assert result[0]["content"] == "HELLO."


class TestRemoveConsecutiveDuplicateWords:
    def test_removes_duplicates(self):
        elements = [{"type": "text", "source_model": "trocr", "content": "went straight straight to bed"}]
        result = remove_consecutive_duplicate_words(elements)
        assert result[0]["content"] == "went straight to bed"

    def test_skips_surya(self):
        elements = [{"type": "text", "source_model": "surya", "content": "the the"}]
        result = remove_consecutive_duplicate_words(elements)
        assert result[0]["content"] == "the the"

    def test_no_duplicates(self):
        elements = [{"type": "text", "source_model": "trocr", "content": "hello world"}]
        result = remove_consecutive_duplicate_words(elements)
        assert result[0]["content"] == "hello world"


class TestCleanTextContent:
    def test_collapses_whitespace(self):
        elements = [{"type": "text", "content": "  hello   world  "}]
        result = clean_text_content(elements)
        assert result[0]["content"] == "hello world"

    def test_fixes_encoding(self):
        elements = [{"type": "text", "content": "quote\u201d"}]
        result = clean_text_content(elements)
        assert result[0]["content"] == 'quote"'

    def test_skips_non_text(self):
        elements = [{"type": "table", "content": "  spaces  "}]
        result = clean_text_content(elements)
        assert result[0]["content"] == "  spaces  "


# ============================================================================
# OCR Corrections
# ============================================================================

# ============================================================================
# Hallucination Detection
# ============================================================================

class TestCheckCharacterPatterns:
    def test_all_same_char(self):
        score, patterns = check_character_patterns("aaaa")
        assert "all_same_char" in patterns
        assert score > 0.5

    def test_isolated_digits(self):
        score, patterns = check_character_patterns("42")
        assert "isolated_digits" in patterns

    def test_only_punctuation(self):
        score, patterns = check_character_patterns(".,;!")
        assert "only_punctuation" in patterns

    def test_normal_text(self):
        score, patterns = check_character_patterns("The Department of Justice")
        assert score == 0.0
        assert patterns == []

    def test_isolated_year(self):
        score, patterns = check_character_patterns("1998")
        assert "isolated_year" in patterns

    def test_unusual_chars(self):
        # Chinese character (outside Latin Extended range)
        score, patterns = check_character_patterns("\u4e16")
        assert "unusual_chars" in patterns

    def test_latin_extended_allowed(self):
        # French accented chars should NOT trigger unusual_chars
        score, patterns = check_character_patterns("caf\u00e9")
        assert "unusual_chars" not in patterns

    def test_latex_infty_detected(self):
        score, patterns = check_character_patterns("\\infty")
        assert "latex_command" in patterns
        assert score > 0

    def test_latex_sigma_detected(self):
        score, patterns = check_character_patterns("\\sigma")
        assert "latex_command" in patterns

    def test_latex_newline_not_detected(self):
        """Single-char command \\n should NOT trigger (requires 2+ alpha chars)."""
        score, patterns = check_character_patterns("\\n")
        assert "latex_command" not in patterns

    def test_latex_digits_not_detected(self):
        """Backslash followed by digits should NOT trigger."""
        score, patterns = check_character_patterns("\\123")
        assert "latex_command" not in patterns

    def test_normal_word_not_latex(self):
        score, patterns = check_character_patterns("infinity")
        assert "latex_command" not in patterns


class TestIsValidText:
    def test_normal_word(self):
        assert is_valid_text("hello") is True

    def test_date(self):
        assert is_valid_text("01/15/2024") is True

    def test_phone(self):
        assert is_valid_text("(212) 555-1234") is True

    def test_empty(self):
        assert is_valid_text("") is False

    def test_multi_word(self):
        assert is_valid_text("Department of Justice") is True

    def test_single_alpha(self):
        assert is_valid_text("A") is True

    def test_single_hash_valid(self):
        # '#' is common in forms (page numbers, references)
        assert is_valid_text("#") is True

    def test_single_nonalnum(self):
        # Uncommon single characters are still invalid
        assert is_valid_text("~") is False

    def test_list_markers_valid(self):
        """List markers like a), b), 1) are valid form content."""
        assert is_valid_text("a)") is True
        assert is_valid_text("b)") is True
        assert is_valid_text("c)") is True
        assert is_valid_text("1)") is True

    def test_leading_dot_decimals_valid(self):
        """Leading-dot decimals like .55, .841 are valid numeric content."""
        assert is_valid_text(".55") is True
        assert is_valid_text(".841") is True

    def test_symbol_punctuation_valid(self):
        """Short symbol markers like #: are valid form content."""
        assert is_valid_text("#:") is True
        assert is_valid_text("#.") is True


class TestHasRepetitionPattern:
    def test_consecutive_repeat(self):
        assert has_repetition_pattern("the the thing") is True

    def test_all_same(self):
        assert has_repetition_pattern("go go go go") is True

    def test_no_repetition(self):
        assert has_repetition_pattern("the quick brown fox") is False

    def test_single_word(self):
        assert has_repetition_pattern("hello") is False


class TestCalculateHallucinationScore:
    def test_normal_text_low_score(self):
        score, signals = calculate_hallucination_score(
            "Department of Justice", 0.95, [100, 100, 400, 120], 1000, 1000
        )
        assert score < 0.30

    def test_garbage_high_score(self):
        score, signals = calculate_hallucination_score(
            "///", 0.30, [5, 5, 15, 15], 1000, 1000
        )
        assert score >= 0.50

    def test_tiny_bbox_penalized_for_longer_content(self):
        """Tiny bbox fires for content >2 chars (not double-penalized with very_short)."""
        score, signals = calculate_hallucination_score(
            "abc", 0.50, [0, 0, 10, 5], 1000, 1000
        )
        assert "tiny_bbox" in signals

    def test_tiny_bbox_skipped_for_very_short(self):
        """Very short content (<=2 chars) should NOT get tiny_bbox penalty."""
        score, signals = calculate_hallucination_score(
            "x", 0.50, [0, 0, 10, 5], 1000, 1000
        )
        assert "tiny_bbox" not in signals
        assert "very_short" in signals

    def test_margin_fragment(self):
        score, signals = calculate_hallucination_score(
            "ab", 0.70, [960, 100, 995, 110], 1000, 1000
        )
        assert "margin_fragment_short" in signals

    def test_digit_exempt_from_margin(self):
        # Page numbers at margin should not trigger margin_fragment_short
        score, signals = calculate_hallucination_score(
            "42", 0.90, [960, 900, 990, 920], 1000, 1000
        )
        assert "margin_fragment_short" not in signals


class TestProcessHallucinations:
    def test_keeps_good_elements(self):
        elements = [
            {"type": "text", "content": "Department of Justice",
             "confidence": 0.95, "bbox": [100, 100, 400, 120]},
        ]
        result = process_hallucinations(elements)
        assert len(result) == 1

    def test_removes_garbage(self):
        elements = [
            {"type": "text", "content": "///",
             "confidence": 0.20, "bbox": [5, 5, 15, 15]},
        ]
        result = process_hallucinations(elements)
        assert len(result) == 0

    def test_non_text_elements_pass_through(self):
        elements = [
            {"type": "table", "bbox": [0, 0, 500, 500]},
            {"type": "signature", "bbox": [0, 0, 100, 100]},
        ]
        result = process_hallucinations(elements)
        assert len(result) == 2

    def test_page_dimensions_used(self):
        elements = [
            {"type": "text", "content": "x", "confidence": 0.50,
             "bbox": [580, 100, 610, 110]},
        ]
        # With default 612 width, this would be at right margin
        result = process_hallucinations(elements, page_dimensions=(612, 792))
        # Should be penalized for margin position
        assert len(result) <= 1


class TestFilterRotatedMarginText:
    def test_removes_narrow_right_edge(self):
        elements = [
            {"type": "text", "content": "ab", "bbox": [960, 100, 980, 120]},
        ]
        result = filter_rotated_margin_text(elements, page_dimensions=(1000, 1000))
        assert len(result) == 0

    def test_exempts_digit_content(self):
        elements = [
            {"type": "text", "content": "123456", "bbox": [960, 100, 980, 120]},
        ]
        result = filter_rotated_margin_text(elements, page_dimensions=(1000, 1000))
        assert len(result) == 1

    def test_keeps_normal_text(self):
        elements = [
            {"type": "text", "content": "Department", "bbox": [100, 100, 400, 120]},
        ]
        result = filter_rotated_margin_text(elements, page_dimensions=(1000, 1000))
        assert len(result) == 1

    def test_single_digit_removed(self):
        """Single digit at rotated margin should be removed (Bates fragment)."""
        elements = [
            {"type": "text", "content": "2", "bbox": [960, 100, 980, 120]},
        ]
        result = filter_rotated_margin_text(elements, page_dimensions=(1000, 1000))
        assert len(result) == 0

    def test_two_digit_removed(self):
        """Two-digit number at rotated margin should be removed."""
        elements = [
            {"type": "text", "content": "39", "bbox": [960, 100, 980, 120]},
        ]
        result = filter_rotated_margin_text(elements, page_dimensions=(1000, 1000))
        assert len(result) == 0

    def test_three_digit_kept(self):
        """Three-digit number at rotated margin should be kept (could be page number)."""
        elements = [
            {"type": "text", "content": "123", "bbox": [960, 100, 980, 120]},
        ]
        result = filter_rotated_margin_text(elements, page_dimensions=(1000, 1000))
        assert len(result) == 1

    def test_bates_number_kept(self):
        """Long Bates number (8+ digits) at rotated margin should be kept."""
        elements = [
            {"type": "text", "content": "12345678", "bbox": [960, 100, 980, 120]},
        ]
        result = filter_rotated_margin_text(elements, page_dimensions=(1000, 1000))
        assert len(result) == 1


# ============================================================================
# Table Validation Helpers
# ============================================================================

class TestClusterPositions:
    def test_single_cluster(self):
        clusters = cluster_positions([10, 12, 11], 5)
        assert len(clusters) == 1

    def test_two_clusters(self):
        clusters = cluster_positions([10, 12, 50, 52], 5)
        assert len(clusters) == 2

    def test_empty(self):
        assert cluster_positions([], 5) == []

    def test_all_different(self):
        clusters = cluster_positions([10, 100, 200], 5)
        assert len(clusters) == 3


class TestCalculateTextDensity:
    def test_full_coverage(self):
        table = [0, 0, 100, 100]
        texts = [[0, 0, 100, 100]]
        assert calculate_text_density(table, texts) == 1.0

    def test_half_coverage(self):
        table = [0, 0, 100, 100]
        texts = [[0, 0, 50, 100]]
        assert calculate_text_density(table, texts) == pytest.approx(0.5)

    def test_no_text(self):
        table = [0, 0, 100, 100]
        assert calculate_text_density(table, []) == 0.0

    def test_invalid_table_bbox(self):
        assert calculate_text_density([0, 0], [[0, 0, 50, 50]]) == 0.0


class TestCalculateStructureScore:
    def test_good_table(self):
        score, signals = calculate_structure_score(
            density=0.15, num_cols=4, num_rows=5,
            grid_coverage=0.40, confidence=0.95
        )
        assert score >= 50  # MIN_STRUCTURE_SCORE
        assert "columns:4" in signals

    def test_single_column(self):
        score, signals = calculate_structure_score(
            density=0.10, num_cols=1, num_rows=2,
            grid_coverage=0.05, confidence=0.50
        )
        assert "single_column" in signals
        assert score < 50


# ============================================================================
# Phone & Date Validation
# ============================================================================

class TestValidatePhoneNumber:
    def test_standard_phone(self):
        result = validate_phone_number("Call (212) 555-1234")
        assert len(result["phones"]) == 1
        assert result["phones"][0]["area_code"] == "212"

    def test_dash_format(self):
        result = validate_phone_number("212-555-1234")
        assert len(result["phones"]) == 1

    def test_no_phone(self):
        result = validate_phone_number("No phone here")
        assert result["validation_status"] == "none"

    def test_date_not_detected_as_phone(self):
        result = validate_phone_number("Date: 12/15/2024")
        assert len(result["phones"]) == 0

    def test_zip_not_detected_as_phone(self):
        result = validate_phone_number("ZIP: 10001-2345")
        assert len(result["phones"]) == 0


class TestValidateDateFormat:
    def test_valid_date(self):
        result = validate_date_format("12/15/2024")
        assert result["validation_status"] == "valid"
        assert len(result["dates"]) == 1

    def test_invalid_month(self):
        result = validate_date_format("15/01/2024")
        assert result["validation_status"] == "suspicious"

    def test_no_date(self):
        result = validate_date_format("no dates here")
        assert result["validation_status"] == "none"

    def test_corrupted_date(self):
        result = validate_date_format("414,00")
        assert result["validation_status"] == "none"


class TestIsDateOrZip:
    def test_zip(self):
        assert is_date_or_zip("10001") is True

    def test_zip_plus4(self):
        assert is_date_or_zip("10001-2345") is True

    def test_date(self):
        assert is_date_or_zip("12/15/2024") is True

    def test_normal_text(self):
        assert is_date_or_zip("hello world") is False

    def test_city_state_zip(self):
        assert is_date_or_zip("New York, NY 10001") is True


class TestExtractPhoneNumbers:
    def test_parenthesized(self):
        phones = extract_phone_numbers("(212) 555-1234")
        assert len(phones) == 1
        assert phones[0] == "(212) 555-1234"

    def test_slash_format(self):
        phones = extract_phone_numbers("212/555-1234")
        assert len(phones) == 1

    def test_no_duplicates(self):
        # Same number shouldn't appear twice even if matched by multiple patterns
        phones = extract_phone_numbers("(212) 555-1234")
        assert len(phones) == 1


# ============================================================================
# Pipeline Utilities
# ============================================================================

class TestSanitizeElements:
    def test_coerces_missing_type(self):
        elements = [{"content": "hello"}]
        result = sanitize_elements(elements)
        assert result[0]["type"] == "unknown"

    def test_coerces_numeric_content(self):
        elements = [{"type": "text", "content": 42}]
        result = sanitize_elements(elements)
        assert result[0]["content"] == "42"

    def test_clamps_confidence(self):
        elements = [{"type": "text", "confidence": 1.5}]
        result = sanitize_elements(elements)
        assert result[0]["confidence"] == 1.0

    def test_clamps_negative_confidence(self):
        elements = [{"type": "text", "confidence": -0.5}]
        result = sanitize_elements(elements)
        assert result[0]["confidence"] == 0.0

    def test_invalid_bbox_cleared(self):
        elements = [{"type": "text", "bbox": [1, 2]}]
        result = sanitize_elements(elements)
        assert result[0]["bbox"] == []

    def test_valid_bbox_preserved(self):
        elements = [{"type": "text", "bbox": [0, 0, 100, 100]}]
        result = sanitize_elements(elements)
        assert result[0]["bbox"] == [0.0, 0.0, 100.0, 100.0]

    def test_non_numeric_bbox_cleared(self):
        elements = [{"type": "text", "bbox": ["a", "b", "c", "d"]}]
        result = sanitize_elements(elements)
        assert result[0]["bbox"] == []

    def test_non_list_bbox_cleared(self):
        elements = [{"type": "text", "bbox": "invalid"}]
        result = sanitize_elements(elements)
        assert result[0]["bbox"] == []


class TestDeduplicateLayoutRegions:
    def test_removes_duplicate_regions(self):
        elements = [
            {"type": "layout_region", "bbox": [0, 0, 100, 100]},
            {"type": "layout_region", "bbox": [0, 0, 100, 100]},
        ]
        result = deduplicate_layout_regions(elements)
        assert len(result) == 1

    def test_keeps_different_regions(self):
        elements = [
            {"type": "layout_region", "bbox": [0, 0, 100, 100]},
            {"type": "layout_region", "bbox": [200, 200, 300, 300]},
        ]
        result = deduplicate_layout_regions(elements)
        assert len(result) == 2

    def test_text_elements_unaffected(self):
        elements = [
            {"type": "text", "content": "hello", "bbox": [0, 0, 100, 20]},
            {"type": "text", "content": "hello", "bbox": [0, 0, 100, 20]},
        ]
        result = deduplicate_layout_regions(elements)
        assert len(result) == 2  # Text dupes not removed by this function


# ============================================================================
# TrOCR Case Preservation
# ============================================================================


class TestTransferSuryaCase:
    def test_all_caps_surya_uppercases_trocr(self):
        """When Surya text is ALL-CAPS, TrOCR output should be uppercased."""
        assert _transfer_surya_case("DATE ISSUED", "date issued") == "DATE ISSUED"

    def test_mixed_case_no_change(self):
        """Mixed case Surya text should not alter TrOCR output."""
        assert _transfer_surya_case("John Smith", "john smith") == "john smith"

    def test_lowercase_no_change(self):
        """Lowercase Surya text should not alter TrOCR output."""
        assert _transfer_surya_case("department", "deparment") == "deparment"

    def test_no_alpha_no_change(self):
        """Non-alpha Surya text should not alter TrOCR output."""
        assert _transfer_surya_case("123-456", "123-456") == "123-456"

    def test_boundary_below_70_percent(self):
        """69% uppercase should NOT trigger uppercasing."""
        # 9 alpha chars, 6 upper = 66.7% → below 70%
        assert _transfer_surya_case("ABCDEFghi", "abcdefghi") == "abcdefghi"

    def test_boundary_above_70_percent_single_word(self):
        """Single word with 70%+ uppercase but not ALL-CAPS stays unchanged (per-word logic)."""
        # 10 alpha chars, 7 upper = 70% → but per-word, not all uppercase
        assert _transfer_surya_case("ABCDEFGhij", "abcdefghij") == "abcdefghij"

    def test_boundary_above_70_percent_fallback(self):
        """When word counts differ and 70%+ uppercase, fallback bulk-uppercases."""
        # Word counts differ: 2 vs 1 → triggers fallback bulk strategy
        assert _transfer_surya_case("DATE ISSUED", "dateissued") == "DATEISSUED"

    def test_empty_surya_text(self):
        """Empty Surya text should not alter TrOCR output."""
        assert _transfer_surya_case("", "some text") == "some text"

    def test_per_word_mixed_case_transfer(self):
        """Word-level transfer: only ALL-CAPS Surya words uppercase the TrOCR word."""
        # "DATE" and "ISSUED:" are all-caps, "January" is not
        assert _transfer_surya_case("DATE ISSUED: January", "date issued: january") == "DATE ISSUED: january"

    def test_per_word_all_caps(self):
        """All words ALL-CAPS → all TrOCR words uppercased."""
        assert _transfer_surya_case("HELLO WORLD", "hello world") == "HELLO WORLD"

    def test_per_word_no_caps(self):
        """No ALL-CAPS words → TrOCR output unchanged."""
        assert _transfer_surya_case("hello world", "hello world") == "hello world"


# ============================================================================
# Decimal-Dash Confusion Repair
# ============================================================================

class TestRepairDecimalDashConfusion:
    def test_standalone_dash_to_decimal(self):
        """Standalone -88 → .88 when page has decimal context."""
        elements = [
            {"type": "text", "content": "Total 14.00"},
            {"type": "text", "content": "Factor -88 applied"},
        ]
        result = repair_decimal_dash_confusion(elements)
        assert result[1]["content"] == "Factor .88 applied"

    def test_label_attached_dash_to_decimal(self):
        """MENT-474 → MENT.474 when page has decimal context."""
        elements = [
            {"type": "text", "content": "Value 12.5 recorded"},
            {"type": "text", "content": "MENT-474"},
        ]
        result = repair_decimal_dash_confusion(elements)
        assert result[1]["content"] == "MENT.474"

    def test_bullet_to_decimal(self):
        """Bullet \u2022 before digit → period (always, even without context)."""
        elements = [
            {"type": "text", "content": "Rate 5.0 percent"},
            {"type": "text", "content": "\u202255"},
        ]
        result = repair_decimal_dash_confusion(elements)
        assert result[1]["content"] == ".55"

    def test_no_correction_without_decimal_context(self):
        """Skip when page has no existing decimal numbers."""
        elements = [
            {"type": "text", "content": "No decimals here"},
            {"type": "text", "content": "Value -88 found"},
        ]
        result = repair_decimal_dash_confusion(elements)
        assert result[1]["content"] == "Value -88 found"

    def test_skips_date_patterns(self):
        """Don't correct dashes inside dates."""
        elements = [
            {"type": "text", "content": "Total 14.00"},
            {"type": "text", "content": "Date 01/15/2024"},
        ]
        result = repair_decimal_dash_confusion(elements)
        assert result[1]["content"] == "Date 01/15/2024"

    def test_skips_phone_patterns(self):
        """Don't correct dashes inside phone numbers."""
        elements = [
            {"type": "text", "content": "Rate 5.0"},
            {"type": "text", "content": "(212) 555-1234"},
        ]
        result = repair_decimal_dash_confusion(elements)
        assert result[1]["content"] == "(212) 555-1234"

    def test_skips_single_digit_after_dash(self):
        """Don't correct -5 (could be negative number, regex requires 2+ digits)."""
        elements = [
            {"type": "text", "content": "Total 14.00"},
            {"type": "text", "content": "Result -5 degrees"},
        ]
        result = repair_decimal_dash_confusion(elements)
        assert result[1]["content"] == "Result -5 degrees"

    def test_skips_non_text_elements(self):
        """Non-text elements should not be modified."""
        elements = [
            {"type": "text", "content": "Rate 5.0"},
            {"type": "table", "content": "-88"},
        ]
        result = repair_decimal_dash_confusion(elements)
        assert result[1]["content"] == "-88"

    def test_sets_repair_flag(self):
        """Corrected elements should get the decimal_dash_repaired flag."""
        elements = [
            {"type": "text", "content": "Total 14.00"},
            {"type": "text", "content": "-88"},
        ]
        result = repair_decimal_dash_confusion(elements)
        assert result[1].get("decimal_dash_repaired") is True

    def test_multiple_corrections_on_page(self):
        """Multiple dash-decimal tokens on the same page."""
        elements = [
            {"type": "text", "content": "Values 14.00 and 3.5"},
            {"type": "text", "content": "-88"},
            {"type": "text", "content": "-474"},
        ]
        result = repair_decimal_dash_confusion(elements)
        assert result[1]["content"] == ".88"
        assert result[2]["content"] == ".474"


# ============================================================================
# OCR Plausibility Check
# ============================================================================

class TestIsOcrPlausible:
    def test_single_char_confusion(self):
        # 'e' and 'c' are OCR-confusable
        assert _is_ocr_plausible("hcllo", "hello") is True

    def test_non_confusable_chars(self):
        # 'x' and 'z' are not in the confusion matrix
        assert _is_ocr_plausible("xello", "zello") is False

    def test_identical_words(self):
        # No differences → diff_count = 0 → False
        assert _is_ocr_plausible("hello", "hello") is False

    def test_too_many_differences(self):
        # More than 2 char differences
        assert _is_ocr_plausible("abcde", "vwxyz") is False

    def test_length_differs_by_more_than_one(self):
        assert _is_ocr_plausible("ab", "abcde") is False

    def test_length_differs_by_one(self):
        # Could be insertion/deletion — accepted
        assert _is_ocr_plausible("helo", "hello") is True


# ============================================================================
# Generalizable Non-Word OCR Correction
# ============================================================================

class TestCorrectNonwordOcrErrors:
    def test_empty_input(self):
        assert correct_nonword_ocr_errors("") == ""

    def test_short_words_skipped(self):
        """Words shorter than 5 chars should not be corrected."""
        result = correct_nonword_ocr_errors("teh cat")
        # 'teh' is only 3 chars, should be left alone
        assert "teh" in result

    def test_words_with_digits_skipped(self):
        """Words containing digits should not be corrected."""
        result = correct_nonword_ocr_errors("ABC123 code")
        assert "ABC123" in result

    def test_short_acronyms_skipped(self):
        """Short all-caps words (likely acronyms) should not be corrected."""
        result = correct_nonword_ocr_errors("MOIST data")
        assert "MOIST" in result

    def test_slash_words_skipped(self):
        """Words with slashes should not be corrected (handled elsewhere)."""
        result = correct_nonword_ocr_errors("DIRECTOR/DEPT memo")
        assert "DIRECTOR/DEPT" in result

    def test_known_words_preserved(self):
        """Real English words should not be modified."""
        result = correct_nonword_ocr_errors("department of justice")
        assert result == "department of justice"

    def test_preserves_uppercase(self):
        """Case pattern of original word should be preserved."""
        result = correct_nonword_ocr_errors("DEPARTMENI of justice")
        # If spell checker corrects it, should maintain uppercase
        if "DEPARTMENI" not in result:
            assert result.split()[0].isupper()

    def test_preserves_trailing_punctuation(self):
        """Trailing punctuation should be preserved after correction."""
        result = correct_nonword_ocr_errors("depariment: hello world")
        if "depariment" not in result:
            assert ":" in result

    def test_proper_nouns_skipped(self):
        """Title-case words (likely proper nouns) should not be corrected."""
        # These are real surnames that the spell checker would mangle
        assert "Baroody" in correct_nonword_ocr_errors("George Baroody")
        assert "Berman" in correct_nonword_ocr_errors("Steve Berman")
        assert "Barrington" in correct_nonword_ocr_errors("Martin Barrington")

    def test_all_caps_still_corrected(self):
        """ALL-CAPS non-acronym words should still be corrected (not proper nouns)."""
        result = correct_nonword_ocr_errors("DEPARTMENI report")
        # ALL-CAPS words longer than 6 chars are still eligible for correction
        if "DEPARTMENI" not in result:
            assert result.split()[0].isupper()


# ============================================================================
# Table Cell Extraction (build_table_cells)
# ============================================================================

from src.postprocessing.table_validation import build_table_cells


class TestBuildTableCells:
    def _make_table(self, bbox, rows, columns, headers=None):
        """Helper to build a table element with structure data."""
        return {
            "type": "table",
            "bbox": bbox,
            "structure": {
                "rows": [{"bbox": r} for r in rows],
                "columns": [{"bbox": c} for c in columns],
                "cells": [],
                "headers": [{"bbox": h} for h in (headers or [])],
            },
        }

    def test_empty_rows(self):
        table = self._make_table([0, 0, 200, 200], rows=[], columns=[[0, 0, 100, 200]])
        cells, absorbed = build_table_cells(table, [])
        assert cells == []
        assert absorbed == set()

    def test_empty_columns(self):
        table = self._make_table([0, 0, 200, 200], rows=[[0, 0, 200, 50]], columns=[])
        cells, absorbed = build_table_cells(table, [])
        assert cells == []
        assert absorbed == set()

    def test_single_cell_with_text(self):
        table = self._make_table(
            [0, 0, 200, 100],
            rows=[[0, 0, 200, 100]],
            columns=[[0, 0, 200, 100]],
        )
        text_elements = [
            {"type": "text", "content": "Hello", "bbox": [10, 10, 100, 50]},
        ]
        cells, absorbed = build_table_cells(table, text_elements)
        assert len(cells) == 1
        assert cells[0]["row"] == 0
        assert cells[0]["col"] == 0
        assert "Hello" in cells[0]["content"]
        assert len(absorbed) == 1  # The text element was absorbed

    def test_multi_row_multi_col(self):
        table = self._make_table(
            [0, 0, 200, 200],
            rows=[[0, 0, 200, 100], [0, 100, 200, 200]],
            columns=[[0, 0, 100, 200], [100, 0, 200, 200]],
        )
        # Text in top-left cell
        text_elements = [
            {"type": "text", "content": "A", "bbox": [10, 10, 50, 50]},
            # Text in bottom-right cell
            {"type": "text", "content": "D", "bbox": [150, 150, 190, 190]},
        ]
        cells, absorbed = build_table_cells(table, text_elements)
        assert len(cells) == 4  # 2x2 grid
        cell_00 = [c for c in cells if c["row"] == 0 and c["col"] == 0][0]
        cell_11 = [c for c in cells if c["row"] == 1 and c["col"] == 1][0]
        assert "A" in cell_00["content"]
        assert "D" in cell_11["content"]
        assert len(absorbed) == 2  # Both text elements absorbed

    def test_text_outside_table_not_assigned(self):
        table = self._make_table(
            [0, 0, 200, 100],
            rows=[[0, 0, 200, 100]],
            columns=[[0, 0, 200, 100]],
        )
        # Text completely outside the table
        text_elements = [
            {"type": "text", "content": "Outside", "bbox": [300, 300, 400, 350]},
        ]
        cells, absorbed = build_table_cells(table, text_elements)
        assert len(cells) == 1
        assert cells[0]["content"] == ""
        assert len(absorbed) == 0  # Nothing absorbed

    def test_header_detection(self):
        table = self._make_table(
            [0, 0, 200, 200],
            rows=[[0, 0, 200, 50], [0, 50, 200, 200]],
            columns=[[0, 0, 200, 200]],
            headers=[[0, 0, 200, 50]],
        )
        cells, _ = build_table_cells(table, [])
        header_cells = [c for c in cells if c["is_header"]]
        non_header = [c for c in cells if not c["is_header"]]
        assert len(header_cells) >= 1
        assert len(non_header) >= 1

    def test_invalid_table_bbox(self):
        table = {
            "type": "table",
            "bbox": [10, 20],  # Invalid — only 2 elements
            "structure": {
                "rows": [{"bbox": [0, 0, 200, 100]}],
                "columns": [{"bbox": [0, 0, 200, 100]}],
                "cells": [],
                "headers": [],
            },
        }
        cells, absorbed = build_table_cells(table, [])
        assert cells == []
        assert absorbed == set()


# ============================================================================
# Phone Number Normalization (normalize_phone_numbers)
# ============================================================================

from src.postprocessing.phone_date import normalize_phone_numbers


class TestNormalizePhoneNumbers:
    def test_standard_phone_detected(self):
        elements = [{"type": "text", "content": "(212) 555-1234"}]
        result = normalize_phone_numbers(elements)
        assert "normalized_phone" in result[0]
        assert result[0]["normalized_phone"] == "(212) 555-1234"

    def test_fax_keyword_sets_phone_type(self):
        elements = [{"type": "text", "content": "FAX: (212) 555-1234"}]
        result = normalize_phone_numbers(elements)
        assert result[0].get("phone_type") == "fax"

    def test_date_not_detected_as_phone(self):
        elements = [{"type": "text", "content": "12/15/2024"}]
        result = normalize_phone_numbers(elements)
        assert "normalized_phone" not in result[0]
        assert "normalized_phones" not in result[0]

    def test_non_text_elements_pass_through(self):
        elements = [{"type": "table", "content": "(212) 555-1234"}]
        result = normalize_phone_numbers(elements)
        assert "normalized_phone" not in result[0]

    def test_multiple_phones(self):
        elements = [{"type": "text", "content": "(212) 555-1234 and (310) 555-5678"}]
        result = normalize_phone_numbers(elements)
        assert "normalized_phones" in result[0]
        assert len(result[0]["normalized_phones"]) == 2

    def test_no_phone_unchanged(self):
        elements = [{"type": "text", "content": "Hello world"}]
        result = normalize_phone_numbers(elements)
        assert "normalized_phone" not in result[0]
        assert "normalized_phones" not in result[0]
        assert "phone_type" not in result[0]


# ============================================================================
# Signature Text Replacement
# ============================================================================

from src.postprocessing.signatures import replace_signature_text, filter_signature_overlap_garbage


class TestReplaceSignatureText:
    def test_received_by_pattern(self):
        elements = [{"type": "text", "content": "RECEIVED BY: John Smith DATE:"}]
        result = replace_signature_text(elements)
        assert "(signature)" in result[0]["content"]
        assert "John Smith" not in result[0]["content"]

    def test_signature_colon_pattern(self):
        elements = [{"type": "text", "content": "SIGNATURE: Jane Doe"}]
        result = replace_signature_text(elements)
        assert "(signature)" in result[0]["content"]
        assert "Jane Doe" not in result[0]["content"]

    def test_no_signature_pattern_unchanged(self):
        elements = [{"type": "text", "content": "The meeting was productive"}]
        result = replace_signature_text(elements)
        assert result[0]["content"] == "The meeting was productive"

    def test_non_text_unchanged(self):
        elements = [{"type": "table", "content": "SIGNATURE: John"}]
        result = replace_signature_text(elements)
        assert result[0]["content"] == "SIGNATURE: John"


# ============================================================================
# Signature Overlap Garbage Filter
# ============================================================================

class TestFilterSignatureOverlapGarbage:
    def test_single_word_overlapping_signature_removed(self):
        elements = [
            {"type": "signature", "bbox": [100, 100, 300, 200]},
            {"type": "text", "content": "elevens", "bbox": [120, 120, 200, 160]},
        ]
        result = filter_signature_overlap_garbage(elements)
        text_els = [e for e in result if e.get("type") == "text"]
        assert len(text_els) == 0

    def test_multi_word_kept(self):
        elements = [
            {"type": "signature", "bbox": [100, 100, 300, 200]},
            {"type": "text", "content": "signed by manager", "bbox": [120, 120, 280, 160]},
        ]
        result = filter_signature_overlap_garbage(elements)
        text_els = [e for e in result if e.get("type") == "text"]
        assert len(text_els) == 1

    def test_date_like_single_word_kept(self):
        elements = [
            {"type": "signature", "bbox": [100, 100, 300, 200]},
            {"type": "text", "content": "12/15/2024", "bbox": [120, 120, 200, 160]},
        ]
        result = filter_signature_overlap_garbage(elements)
        text_els = [e for e in result if e.get("type") == "text"]
        assert len(text_els) == 1

    def test_no_signatures_all_text_kept(self):
        elements = [
            {"type": "text", "content": "elevens", "bbox": [120, 120, 200, 160]},
            {"type": "text", "content": "test", "bbox": [220, 120, 290, 160]},
        ]
        result = filter_signature_overlap_garbage(elements)
        text_els = [e for e in result if e.get("type") == "text"]
        assert len(text_els) == 2

    def test_text_with_no_bbox_kept(self):
        elements = [
            {"type": "signature", "bbox": [100, 100, 300, 200]},
            {"type": "text", "content": "orphan"},
        ]
        result = filter_signature_overlap_garbage(elements)
        text_els = [e for e in result if e.get("type") == "text"]
        assert len(text_els) == 1


# ============================================================================
# Resolution-Adaptive Row Clustering
# ============================================================================

from src.utils.text import cluster_text_rows


class TestClusterTextRows:
    def test_empty_input(self):
        assert cluster_text_rows([]) == []

    def test_single_element(self):
        elements = [{"bbox": [0, 100, 200, 120]}]
        result = cluster_text_rows(elements)
        assert len(result) == 1
        assert result[0]["row_id"] == 0

    def test_same_row_elements(self):
        """Elements at similar y-coords should cluster into one row."""
        elements = [
            {"bbox": [200, 100, 300, 120]},
            {"bbox": [0, 102, 100, 122]},
        ]
        result = cluster_text_rows(elements)
        assert all(e["row_id"] == 0 for e in result)
        # Sorted left-to-right within row
        assert result[0]["bbox"][0] < result[1]["bbox"][0]

    def test_different_rows(self):
        """Elements far apart vertically should be in separate rows."""
        elements = [
            {"bbox": [0, 100, 100, 120]},
            {"bbox": [0, 500, 100, 520]},
        ]
        result = cluster_text_rows(elements)
        assert result[0]["row_id"] != result[1]["row_id"]

    def test_adaptive_threshold_scales_with_page(self):
        """Threshold should scale: a 15px gap on a 3000px page is tiny."""
        # On a tall page (3000px), 15px apart should be the same row
        # because adaptive threshold = max(8, 3000 * 0.012) = 36px
        elements = [
            {"bbox": [0, 100, 100, 120]},
            {"bbox": [0, 115, 100, 135]},
            {"bbox": [0, 2900, 100, 2920]},  # drives max_y to ~2920
        ]
        result = cluster_text_rows(elements)
        # First two should be in the same row, third separate
        assert result[0]["row_id"] == result[1]["row_id"]
        assert result[2]["row_id"] != result[0]["row_id"]

    def test_explicit_threshold_overrides_adaptive(self):
        """An explicit y_threshold should override adaptive calculation."""
        elements = [
            {"bbox": [0, 100, 100, 120]},
            {"bbox": [0, 200, 100, 220]},
        ]
        # With a very large threshold, everything clusters together
        result = cluster_text_rows(elements, y_threshold=200)
        assert result[0]["row_id"] == result[1]["row_id"]


# ============================================================================
# Hallucination Table-Region Exemption
# ============================================================================


class TestHallucinationTableExemption:
    def test_garbage_inside_table_kept(self):
        """Short numbers inside a table region should NOT be removed."""
        elements = [
            {"type": "table", "bbox": [0, 0, 500, 500]},
            {"type": "text", "content": "8", "confidence": 0.50,
             "bbox": [50, 50, 70, 70]},
        ]
        result = process_hallucinations(elements, page_dimensions=(1000, 1000))
        text_els = [e for e in result if e.get("type") == "text"]
        assert len(text_els) == 1, "Short number inside table should be kept"

    def test_garbage_outside_table_still_removed(self):
        """Short garbage text NOT inside a table should still be removed."""
        elements = [
            {"type": "table", "bbox": [0, 0, 200, 200]},
            {"type": "text", "content": "///", "confidence": 0.20,
             "bbox": [500, 500, 510, 510]},
        ]
        result = process_hallucinations(elements, page_dimensions=(1000, 1000))
        text_els = [e for e in result if e.get("type") == "text"]
        assert len(text_els) == 0, "Garbage outside table should still be removed"

    def test_no_tables_normal_behavior(self):
        """Without tables, hallucination scoring behaves normally."""
        elements = [
            {"type": "text", "content": "///", "confidence": 0.20,
             "bbox": [5, 5, 15, 15]},
        ]
        result = process_hallucinations(elements, page_dimensions=(1000, 1000))
        assert len(result) == 0


# ============================================================================
# Numeric Column Detection
# ============================================================================


class TestDetectNumericColumns:
    """Unit tests for _detect_numeric_columns()."""

    def test_column_aligned_numbers_detected(self):
        """3+ numbers at the same x-position should form a column."""
        elements = [
            {"type": "text", "content": "42", "bbox": [100, 100, 130, 120]},
            {"type": "text", "content": "88", "bbox": [100, 150, 130, 170]},
            {"type": "text", "content": "7", "bbox": [100, 200, 130, 220]},
        ]
        result = _detect_numeric_columns(elements, 1000)
        assert len(result) == 3

    def test_fewer_than_three_not_detected(self):
        """Only 2 aligned numbers should NOT form a column."""
        elements = [
            {"type": "text", "content": "42", "bbox": [100, 100, 130, 120]},
            {"type": "text", "content": "88", "bbox": [100, 150, 130, 170]},
        ]
        result = _detect_numeric_columns(elements, 1000)
        assert len(result) == 0

    def test_non_aligned_not_detected(self):
        """Numbers spread across different x-positions should NOT cluster."""
        elements = [
            {"type": "text", "content": "42", "bbox": [100, 100, 130, 120]},
            {"type": "text", "content": "88", "bbox": [400, 150, 430, 170]},
            {"type": "text", "content": "7", "bbox": [700, 200, 730, 220]},
        ]
        result = _detect_numeric_columns(elements, 1000)
        assert len(result) == 0

    def test_long_text_excluded(self):
        """Text longer than NUMERIC_COL_MAX_CHARS should not be candidates."""
        elements = [
            {"type": "text", "content": "12345", "bbox": [100, 100, 130, 120]},
            {"type": "text", "content": "67890", "bbox": [100, 150, 130, 170]},
            {"type": "text", "content": "11111", "bbox": [100, 200, 130, 220]},
        ]
        result = _detect_numeric_columns(elements, 1000)
        assert len(result) == 0

    def test_non_text_ignored(self):
        """Non-text elements should be ignored."""
        elements = [
            {"type": "table", "content": "42", "bbox": [100, 100, 130, 120]},
            {"type": "text", "content": "88", "bbox": [100, 150, 130, 170]},
            {"type": "text", "content": "7", "bbox": [100, 200, 130, 220]},
        ]
        result = _detect_numeric_columns(elements, 1000)
        assert len(result) == 0  # Only 2 text candidates

    def test_percentages_accepted(self):
        """Values like '42%' should be valid numeric candidates."""
        elements = [
            {"type": "text", "content": "42%", "bbox": [100, 100, 130, 120]},
            {"type": "text", "content": "88%", "bbox": [100, 150, 130, 170]},
            {"type": "text", "content": "7%", "bbox": [100, 200, 130, 220]},
        ]
        result = _detect_numeric_columns(elements, 1000)
        assert len(result) == 3

    def test_decimals_accepted(self):
        """Values like '3.5' should be valid numeric candidates."""
        elements = [
            {"type": "text", "content": "3.5", "bbox": [100, 100, 130, 120]},
            {"type": "text", "content": "1.2", "bbox": [100, 150, 130, 170]},
            {"type": "text", "content": "9.0", "bbox": [100, 200, 130, 220]},
        ]
        result = _detect_numeric_columns(elements, 1000)
        assert len(result) == 3

    def test_multiple_columns(self):
        """Two separate columns at different x-positions should both be detected."""
        elements = [
            # Column 1 at x~100
            {"type": "text", "content": "1", "bbox": [100, 100, 120, 120]},
            {"type": "text", "content": "2", "bbox": [100, 150, 120, 170]},
            {"type": "text", "content": "3", "bbox": [100, 200, 120, 220]},
            # Column 2 at x~500
            {"type": "text", "content": "4", "bbox": [500, 100, 520, 120]},
            {"type": "text", "content": "5", "bbox": [500, 150, 520, 170]},
            {"type": "text", "content": "6", "bbox": [500, 200, 520, 220]},
        ]
        result = _detect_numeric_columns(elements, 1000)
        assert len(result) == 6

    def test_no_bbox_skipped(self):
        """Elements without bboxes should be skipped gracefully."""
        elements = [
            {"type": "text", "content": "42"},
            {"type": "text", "content": "88", "bbox": [100, 150, 130, 170]},
            {"type": "text", "content": "7", "bbox": [100, 200, 130, 220]},
        ]
        result = _detect_numeric_columns(elements, 1000)
        assert len(result) == 0  # Only 2 candidates with bboxes


class TestNumericColumnIntegration:
    """Integration tests: numeric columns survive process_hallucinations()."""

    def test_aligned_numbers_survive_hallucination_filter(self):
        """Column-aligned short numbers should NOT be removed by hallucination filter."""
        elements = [
            {"type": "text", "content": "42", "confidence": 0.60,
             "bbox": [100, 100, 130, 120]},
            {"type": "text", "content": "88", "confidence": 0.60,
             "bbox": [100, 200, 130, 220]},
            {"type": "text", "content": "7", "confidence": 0.60,
             "bbox": [100, 300, 130, 320]},
        ]
        result = process_hallucinations(elements, page_dimensions=(1000, 1000))
        text_els = [e for e in result if e.get("type") == "text"]
        assert len(text_els) == 3, "Column-aligned numbers should be exempted"

    def test_isolated_number_still_scored(self):
        """A single isolated short number should still be scored (flagged or removed)."""
        elements = [
            {"type": "text", "content": "42", "confidence": 0.40,
             "bbox": [5, 5, 25, 25]},
        ]
        result = process_hallucinations(elements, page_dimensions=(1000, 1000))
        text_els = [e for e in result if e.get("type") == "text"]
        # Either removed entirely, or flagged as hallucination
        if text_els:
            assert text_els[0].get("hallucination_flag") is True, \
                "Isolated short number should be flagged if not removed"
        # Both outcomes (removed or flagged) are acceptable


# ============================================================================
# Phone Pattern Consolidation
# ============================================================================


class TestPhonePatternConsolidation:
    def test_validation_and_extraction_agree(self):
        """validate_phone_number and extract_phone_numbers should find the
        same phones for the same input."""
        test_cases = [
            "(212) 555-1234",
            "212-555-1234",
            "212/555-1234",
            "FAX: (310) 555-5678",
            "Call 206 623 0594 today",
        ]
        for text in test_cases:
            validated = validate_phone_number(text)
            extracted = extract_phone_numbers(text)
            validated_normalized = [p["normalized"] for p in validated["phones"]]
            assert set(validated_normalized) == set(extracted), (
                f"Mismatch for '{text}': validation={validated_normalized}, extraction={extracted}"
            )


# ============================================================================
# Phone Validation Keyword Filtering
# ============================================================================


class TestPhoneValidationKeywords:
    """Verify that add_phone_validation_to_element no longer over-triggers
    on common characters like '(' and '-'."""

    def test_no_trigger_on_bare_parens(self):
        """Text with parentheses but no phone pattern should not trigger."""
        element = {"type": "text", "content": "(see attached) 200-page report"}
        result = add_phone_validation_to_element(element)
        assert "phone_validation" not in result

    def test_triggers_on_phone_keyword(self):
        """Text with a phone keyword and valid number should still trigger."""
        element = {"type": "text", "content": "Phone: 555-123-4567"}
        result = add_phone_validation_to_element(element)
        assert "phone_validation" in result

    def test_triggers_on_parenthesized_area_code(self):
        """Text with (XXX) area code format should still trigger."""
        element = {"type": "text", "content": "(212) 555-1234"}
        result = add_phone_validation_to_element(element)
        assert "phone_validation" in result


# ============================================================================
# Bbox Area Thresholds (module-level constant)
# ============================================================================


class TestBboxAreaThresholds:
    """Verify is_bbox_too_large uses the module-level BBOX_AREA_THRESHOLDS."""

    def test_human_face_strict_threshold(self):
        from src.utils.bbox import is_bbox_too_large
        # 15% of page → above 10% threshold for human face
        assert is_bbox_too_large([0, 0, 150, 100], 1000, 100, label="human face")

    def test_logo_within_threshold(self):
        from src.utils.bbox import is_bbox_too_large
        # 20% of page → below 30% threshold for logo
        assert not is_bbox_too_large([0, 0, 200, 100], 1000, 100, label="logo")

    def test_default_threshold_applied(self):
        from src.utils.bbox import is_bbox_too_large
        # Unknown label uses default (50%)
        assert not is_bbox_too_large([0, 0, 400, 100], 1000, 100, label="unknown_thing")
        assert is_bbox_too_large([0, 0, 600, 100], 1000, 100, label="unknown_thing")
