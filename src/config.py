"""
Centralized configuration for cross-cutting pipeline thresholds.

Thresholds that are referenced by multiple modules or that critically
control pipeline behavior are collected here.  Module-specific constants
(e.g. hallucination signal weights, classification feature weights)
remain in their own files.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineConfig:
    """Key thresholds controlling pipeline behavior.

    Frozen dataclass — treat as read-only at runtime.  To experiment with
    different values, create a new instance and pass it through.
    """

    # ------------------------------------------------------------------
    # Page region boundaries
    # ------------------------------------------------------------------
    # Signature region: everything below this *ratio* of page height.
    # 0.80 → bottom 20% of the page is considered the signature zone.
    # Used by classification (region isolation) and routing (uncertainty flag).
    signature_region_ratio: float = 0.80

    # ------------------------------------------------------------------
    # TrOCR confidence calibration
    # ------------------------------------------------------------------
    # Maps TrOCR's negative log-probability to a 0-1 confidence score:
    #   confidence = clamp(1.0 + log_prob / divisor)
    # Empirically calibrated: TrOCR log-probs on the 32-doc validation set
    # range from roughly -3.0 (very uncertain) to 0.0 (perfect).  A divisor
    # of 2.5 maps that range onto [0, 1] with the 50% crossover at -1.25.
    trocr_confidence_divisor: float = 2.5

    # ------------------------------------------------------------------
    # Post-processing: classification override
    # ------------------------------------------------------------------
    # If this fraction (or more) of text elements were sourced from TrOCR,
    # override the page's document_type to "handwritten".
    trocr_majority_threshold: float = 0.50

    # ------------------------------------------------------------------
    # OCR routing thresholds
    # ------------------------------------------------------------------
    text_confidence_threshold: float = 0.4    # Min Surya confidence to keep a line
    hw_region_min_confidence: float = 0.15    # Lower gate for lines in detected handwriting regions
    doc_handwritten_gate: float = 0.45        # Classification confidence gate for TrOCR-only mode
    typed_low_conf_threshold: float = 0.60    # Below this, typed lines get TrOCR fallback
    signature_trocr_gate: float = 0.90        # Strict confidence gate for signature regions
    trocr_bbox_padding: int = 12              # Pixels of padding for descenders/ascenders

    # ------------------------------------------------------------------
    # Post-processing: hallucination thresholds
    # ------------------------------------------------------------------
    hallucination_remove_threshold: float = 0.50   # Score >= this → remove element
    hallucination_flag_threshold: float = 0.30     # Score > this → flag but keep

    # ------------------------------------------------------------------
    # Post-processing: table validation
    # ------------------------------------------------------------------
    min_structure_score: float = 50    # Minimum score to keep a detected table
    min_overlap_ratio: float = 0.3    # 30% overlap required for region–text association

    # ------------------------------------------------------------------
    # Post-processing: signature overlap garbage filter
    # ------------------------------------------------------------------
    signature_overlap_threshold: float = 0.50  # Overlap ratio to classify text as signature garbage


# Singleton used by all modules.  Import this, not PipelineConfig.
CONFIG = PipelineConfig()
