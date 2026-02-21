"""
Flexible Evaluation Module for Document OCR Pipeline.
Provides both strict and flexible WER/CER calculation with error categorization.
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict


@dataclass
class EvaluationResult:
    """Container for evaluation metrics."""
    strict_wer: float
    flexible_wer: float
    strict_cer: float
    flexible_cer: float
    formatting_errors: int
    content_errors: int
    total_ref_words: int
    total_ref_chars: int


def levenshtein_distance(s1: List, s2: List) -> int:
    """
    Calculate the Levenshtein distance between two sequences.

    Args:
        s1: First sequence (list of tokens or characters)
        s2: Second sequence

    Returns:
        Edit distance (insertions, deletions, substitutions)
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


class FlexibleEvaluator:
    """
    Flexible WER/CER evaluation with configurable normalization.

    Features:
    - Strict and flexible WER/CER calculation
    - Punctuation-insensitive comparison (optional)
    - Consistent tokenization handling combined words
    - Error categorization (formatting vs content)
    """

    # Punctuation that can be ignored in flexible evaluation
    PUNCTUATION = set('.,;:!?\'"-()[]{}/<>@#$%^&*_+=|\\~`')

    def __init__(self, ignore_punctuation: bool = True):
        """
        Initialize evaluator.

        Args:
            ignore_punctuation: If True, flexible metrics ignore punctuation differences
        """
        self.ignore_punctuation = ignore_punctuation

    def tokenize(self, text: str, flexible: bool = False) -> List[str]:
        """
        Tokenize text for WER calculation.

        Handles combined words and normalizes for comparison.

        Args:
            text: Input text
            flexible: If True, apply flexible normalization

        Returns:
            List of word tokens
        """
        # Lowercase for case-insensitive comparison
        text = text.lower()

        if flexible and self.ignore_punctuation:
            # Remove punctuation for flexible comparison
            text = ''.join(c if c not in self.PUNCTUATION else ' ' for c in text)

        # Split words, handling multiple spaces
        words = text.split()

        return words

    def normalize_for_cer(self, text: str, flexible: bool = False) -> str:
        """
        Normalize text for CER calculation.

        Args:
            text: Input text
            flexible: If True, apply flexible normalization

        Returns:
            Normalized character sequence
        """
        # Lowercase
        text = text.lower()

        if flexible and self.ignore_punctuation:
            # Remove punctuation and spaces
            text = ''.join(c for c in text if c not in self.PUNCTUATION and c != ' ')
        else:
            # Only remove spaces for strict CER
            text = text.replace(' ', '')

        return text

    def calculate_strict_wer(self, reference: str, hypothesis: str) -> float:
        """
        Calculate strict Word Error Rate.

        Standard WER with no special handling.

        Args:
            reference: Ground truth text
            hypothesis: OCR output text

        Returns:
            WER as float (0.0 to 1.0+)
        """
        ref_words = self.tokenize(reference, flexible=False)
        hyp_words = self.tokenize(hypothesis, flexible=False)

        if not ref_words:
            return 0.0

        distance = levenshtein_distance(ref_words, hyp_words)
        return distance / len(ref_words)

    def calculate_flexible_wer(self, reference: str, hypothesis: str) -> float:
        """
        Calculate flexible Word Error Rate.

        Ignores punctuation differences when comparing words.

        Args:
            reference: Ground truth text
            hypothesis: OCR output text

        Returns:
            Flexible WER as float
        """
        ref_words = self.tokenize(reference, flexible=True)
        hyp_words = self.tokenize(hypothesis, flexible=True)

        if not ref_words:
            return 0.0

        distance = levenshtein_distance(ref_words, hyp_words)
        return distance / len(ref_words)

    def calculate_strict_cer(self, reference: str, hypothesis: str) -> float:
        """
        Calculate strict Character Error Rate.

        Standard CER with spaces removed.

        Args:
            reference: Ground truth text
            hypothesis: OCR output text

        Returns:
            CER as float
        """
        ref_chars = self.normalize_for_cer(reference, flexible=False)
        hyp_chars = self.normalize_for_cer(hypothesis, flexible=False)

        if not ref_chars:
            return 0.0

        distance = levenshtein_distance(list(ref_chars), list(hyp_chars))
        return distance / len(ref_chars)

    def calculate_flexible_cer(self, reference: str, hypothesis: str) -> float:
        """
        Calculate flexible Character Error Rate.

        Ignores punctuation differences.

        Args:
            reference: Ground truth text
            hypothesis: OCR output text

        Returns:
            Flexible CER as float
        """
        ref_chars = self.normalize_for_cer(reference, flexible=True)
        hyp_chars = self.normalize_for_cer(hypothesis, flexible=True)

        if not ref_chars:
            return 0.0

        distance = levenshtein_distance(list(ref_chars), list(hyp_chars))
        return distance / len(ref_chars)

    def categorize_errors(self, reference: str, hypothesis: str) -> Dict[str, int]:
        """
        Categorize errors as formatting or content errors.

        Formatting errors: Differences in punctuation, spacing, capitalization
        Content errors: Actual word or character differences

        Args:
            reference: Ground truth text
            hypothesis: OCR output text

        Returns:
            Dictionary with 'formatting' and 'content' error counts
        """
        # Compare strict vs flexible WER to estimate formatting errors
        strict_wer = self.calculate_strict_wer(reference, hypothesis)
        flexible_wer = self.calculate_flexible_wer(reference, hypothesis)

        ref_words = self.tokenize(reference, flexible=False)
        total_words = len(ref_words) if ref_words else 1

        # Estimate error counts
        strict_errors = int(strict_wer * total_words)
        flexible_errors = int(flexible_wer * total_words)

        formatting_errors = max(0, strict_errors - flexible_errors)
        content_errors = flexible_errors

        return {
            'formatting': formatting_errors,
            'content': content_errors,
        }

    def evaluate(self, reference: str, hypothesis: str) -> EvaluationResult:
        """
        Run full evaluation suite.

        Calculates all metrics and error categorization.

        Args:
            reference: Ground truth text
            hypothesis: OCR output text

        Returns:
            EvaluationResult with all metrics
        """
        strict_wer = self.calculate_strict_wer(reference, hypothesis)
        flexible_wer = self.calculate_flexible_wer(reference, hypothesis)
        strict_cer = self.calculate_strict_cer(reference, hypothesis)
        flexible_cer = self.calculate_flexible_cer(reference, hypothesis)

        error_cats = self.categorize_errors(reference, hypothesis)

        ref_words = self.tokenize(reference, flexible=False)
        ref_chars = self.normalize_for_cer(reference, flexible=False)

        return EvaluationResult(
            strict_wer=strict_wer,
            flexible_wer=flexible_wer,
            strict_cer=strict_cer,
            flexible_cer=flexible_cer,
            formatting_errors=error_cats['formatting'],
            content_errors=error_cats['content'],
            total_ref_words=len(ref_words),
            total_ref_chars=len(ref_chars),
        )
