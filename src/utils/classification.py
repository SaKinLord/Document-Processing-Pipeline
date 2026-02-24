"""
Document type classification (typed, handwritten, mixed).

Multi-feature analysis using stroke width, line regularity, angle variance,
edge density, form structure, character uniformity, signature isolation,
and fax/letterhead header detection.
"""

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from src.config import CONFIG

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration Dataclasses
# ============================================================================

@dataclass(frozen=True)
class FeatureConfig:
    """Scoring thresholds and weights for a classification feature.

    Used by ``_score_feature`` (higher value = more typed) and
    ``_score_feature_inverted`` (lower value = more typed).

    ``high_thresh``/``high_weight`` correspond to the strongest signal tier,
    ``med_thresh``/``med_weight`` to the moderate tier, and the optional
    ``low_thresh``/``low_weight`` to a weak third tier.
    """
    high_thresh: float
    med_thresh: float
    high_weight: float
    med_weight: float
    low_thresh: float = 0.0
    low_weight: float = 0.0


@dataclass(frozen=True)
class AngleConfig:
    """Angle variance scoring: positional typed signal + handwriting penalties."""
    low_thresh: float = 400
    med_thresh: float = 800
    high_thresh: float = 1500
    med_high_thresh: float = 1200
    form_gate: float = 0.3
    low_weight: float = 0.10
    med_weight: float = 0.05
    high_penalty: float = -0.15
    med_high_penalty: float = -0.08


@dataclass(frozen=True)
class EdgeConfig:
    """Edge density scoring: typed range and sparse fallback."""
    typed_range: tuple = (0.02, 0.15)
    sparse_thresh: float = 0.02
    typed_weight: float = 0.08
    sparse_weight: float = 0.04


@dataclass(frozen=True)
class SignatureDetectionConfig:
    """Parameters for signature region detection via connected components."""
    min_height: int = 20
    min_width: int = 50
    min_ink_ratio: float = 0.003
    max_ink_ratio: float = 0.15
    min_component_area: int = 50
    max_component_ratio: float = 0.3
    min_component_width: int = 20
    min_components: int = 1
    max_components: int = 15


@dataclass(frozen=True)
class FaxHeaderConfig:
    """Parameters for fax/letterhead header detection via morphology."""
    min_height: int = 30
    min_width: int = 100
    min_kernel_w: int = 25
    kernel_divisor: int = 20
    line_coverage_ratio: float = 0.1
    ink_range: tuple = (0.02, 0.25)


@dataclass(frozen=True)
class RuledPaperConfig:
    """Ruled paper handwriting detection overrides."""
    line_thresh: float = 0.95
    form_thresh: float = 0.25
    angle_range: tuple = (300, 700)
    penalty: float = -0.12


@dataclass(frozen=True)
class SparseFormConfig:
    """Sparse/blank form detection — forces typed classification floor."""
    edge_thresh: float = 0.02
    form_thresh: float = 0.15
    floor: float = 0.65


@dataclass(frozen=True)
class ClassificationConfig:
    """Top-level classification configuration grouping all sub-configs."""
    # Feature scoring configs
    stroke: FeatureConfig = FeatureConfig(
        high_thresh=0.3, med_thresh=0.5, high_weight=0.20, med_weight=0.10,
    )
    line_regularity: FeatureConfig = FeatureConfig(
        high_thresh=0.7, med_thresh=0.5, high_weight=0.12, med_weight=0.06,
    )
    angle: AngleConfig = AngleConfig()
    edge: EdgeConfig = EdgeConfig()
    form: FeatureConfig = FeatureConfig(
        high_thresh=0.5, med_thresh=0.2, high_weight=0.30, med_weight=0.20,
        low_thresh=0.1, low_weight=0.12,
    )
    uniformity: FeatureConfig = FeatureConfig(
        high_thresh=0.7, med_thresh=0.4, high_weight=0.12, med_weight=0.06,
    )

    # Detection configs
    signature: SignatureDetectionConfig = SignatureDetectionConfig()
    fax_header: FaxHeaderConfig = FaxHeaderConfig()
    ruled_paper: RuledPaperConfig = RuledPaperConfig()
    sparse_form: SparseFormConfig = SparseFormConfig()

    # Combo scoring weights
    fax_header_weight: float = 0.20
    sig_on_form_thresh: float = 0.2
    sig_on_form_weight: float = 0.10

    # Classification boundaries
    typed_threshold: float = 0.65
    handwritten_threshold: float = 0.45
    max_confidence: float = 0.95

    # Image preprocessing
    max_analysis_width: int = 1000
    header_region_ratio: float = 0.15

    # Feature defaults (returned on analysis failure)
    default_score: float = 0.5
    default_angle_variance: float = 1000.0

    # Stroke analysis parameters
    stroke_min_samples: int = 100
    stroke_min_mean_width: float = 0.1
    stroke_cv_normalizer: float = 0.5

    # Line regularity analysis parameters
    line_min_hough_lines: int = 5
    line_hough_threshold: int = 50
    line_horizontal_angle_tolerance: int = 5

    # Form structure analysis parameters
    form_line_ratio_normalizer: float = 0.01
    form_min_kernel_size: int = 20
    form_kernel_divisor: int = 25

    # Character uniformity analysis parameters
    uniformity_min_components: int = 10
    uniformity_cv_normalizer: float = 0.4


CFG = ClassificationConfig()


# ============================================================================
# Public API
# ============================================================================

def classify_document_type(image):
    """
    Classify document as typed, handwritten, or mixed based on 8 features.

    Returns: tuple (doc_type: str, confidence: float)
        - doc_type: "typed", "handwritten", or "mixed"
        - confidence: 0.0 to 1.0
    """
    try:
        gray = _prepare_grayscale(image)
        h, w = gray.shape

        # Region isolation
        sig_start = int(h * CONFIG.signature_region_ratio)
        main_body = gray[:sig_start, :]
        signature_region = gray[sig_start:, :]
        header_region = gray[:int(h * CFG.header_region_ratio), :]

        has_signature = _detect_signature_in_region(signature_region)
        has_fax_header = _detect_fax_header(header_region)
        analysis_region = main_body if has_signature else gray

        # Feature extraction
        scores = _extract_features(analysis_region, gray)
        typed_score = _compute_typed_score(scores, has_signature, has_fax_header)

        _log_classification(scores, has_signature, has_fax_header, typed_score)
        return _classify_from_score(typed_score)

    except (cv2.error, ValueError, IndexError) as e:
        logger.warning("Document classification failed, defaulting to mixed. Error: %s", e)
        return ("mixed", 0.3)


def detect_signature_region(image):
    """
    Detect if the bottom region of a document contains a handwritten signature.

    Returns:
        dict with keys:
            - 'has_signature': bool
            - 'region_bbox': [x1, y1, x2, y2] normalized to 0-1 range, or None
    """
    try:
        if hasattr(image, 'mode'):
            gray = np.array(image.convert('L'))
        else:
            gray = image

        h, w = gray.shape[:2] if len(gray.shape) >= 2 else (gray.shape[0], 1)

        sig_ratio = CONFIG.signature_region_ratio
        sig_start = int(h * sig_ratio)
        signature_region = gray[sig_start:, :]

        if _detect_signature_in_region(signature_region):
            return {
                'has_signature': True,
                'region_bbox': [0.0, sig_ratio, 1.0, 1.0]
            }
        return {'has_signature': False, 'region_bbox': None}

    except (cv2.error, ValueError, IndexError) as e:
        logger.warning("detect_signature_region failed: %s", e)
        return {'has_signature': False, 'region_bbox': None}


# ============================================================================
# Scoring Helpers
# ============================================================================

def _score_feature(value, cfg):
    """Score a feature where higher value = more typed.

    Returns the weight for the highest tier the value exceeds.
    """
    if value > cfg.high_thresh:
        return cfg.high_weight
    if value > cfg.med_thresh:
        return cfg.med_weight
    if cfg.low_thresh and value > cfg.low_thresh:
        return cfg.low_weight
    return 0.0


def _score_feature_inverted(value, cfg):
    """Score a feature where lower value = more typed.

    Returns the weight for the highest tier the value falls below.
    """
    if value < cfg.high_thresh:
        return cfg.high_weight
    if value < cfg.med_thresh:
        return cfg.med_weight
    return 0.0


# ============================================================================
# Internal Helpers
# ============================================================================

def _prepare_grayscale(image):
    """Convert image to grayscale numpy array, resizing if too wide."""
    img_np = np.array(image)
    if len(img_np.shape) == 3:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_np

    h, w = gray.shape
    if w > CFG.max_analysis_width:
        scale = CFG.max_analysis_width / w
        gray = cv2.resize(gray, (0, 0), fx=scale, fy=scale)

    return gray


def _extract_features(analysis_region, full_page_gray):
    """Extract all 6 numeric features from the analysis region."""
    return {
        'stroke': _calculate_stroke_variance(analysis_region),
        'line': _calculate_line_regularity(analysis_region),
        'angle': _calculate_angle_variance(analysis_region),
        'edge': _calculate_edge_density(analysis_region),
        'form': _detect_form_structure(full_page_gray),
        'uniformity': _calculate_character_uniformity(analysis_region),
    }


def _compute_typed_score(scores, has_signature, has_fax_header):
    """Combine feature scores into a single typed_score using weighted voting."""
    typed_score = 0.0

    # Stroke variance: low = typed
    typed_score += _score_feature_inverted(scores['stroke'], CFG.stroke)

    # Line regularity: high = typed
    typed_score += _score_feature(scores['line'], CFG.line_regularity)

    # Angle variance (custom — penalties + cross-feature gate)
    angle = scores['angle']
    acfg = CFG.angle
    if angle < acfg.low_thresh:
        typed_score += acfg.low_weight
    elif angle < acfg.med_thresh:
        typed_score += acfg.med_weight
    elif angle > acfg.high_thresh and scores['form'] < acfg.form_gate:
        typed_score += acfg.high_penalty
    elif angle > acfg.med_high_thresh and scores['form'] < acfg.form_gate:
        typed_score += acfg.med_high_penalty

    # Edge density (custom — range check)
    edge = scores['edge']
    ecfg = CFG.edge
    if ecfg.typed_range[0] < edge < ecfg.typed_range[1]:
        typed_score += ecfg.typed_weight
    elif edge < ecfg.sparse_thresh:
        typed_score += ecfg.sparse_weight

    # Form structure: high = typed (3 tiers)
    typed_score += _score_feature(scores['form'], CFG.form)

    # Character uniformity: high = typed
    typed_score += _score_feature(scores['uniformity'], CFG.uniformity)

    # Fax/letterhead header boost
    if has_fax_header:
        typed_score += CFG.fax_header_weight

    # Signature on a form
    if has_signature and scores['form'] > CFG.sig_on_form_thresh:
        typed_score += CFG.sig_on_form_weight

    # Ruled paper handwriting penalty
    rp = CFG.ruled_paper
    is_ruled = (
        scores['line'] >= rp.line_thresh
        and scores['form'] < rp.form_thresh
        and rp.angle_range[0] < scores['angle'] < rp.angle_range[1]
    )
    if is_ruled:
        typed_score += rp.penalty
        logger.debug("Ruled paper handwriting detected: %.2f adjustment", rp.penalty)

    # Sparse/blank form floor
    sp = CFG.sparse_form
    is_sparse = (
        scores['edge'] < sp.edge_thresh
        and scores['form'] > sp.form_thresh
        and not is_ruled
    )
    if is_sparse and typed_score < sp.floor:
        typed_score = sp.floor
        logger.debug("Sparse/blank form detected: forcing typed classification")

    return max(0.0, min(typed_score, 1.0))


def _classify_from_score(typed_score):
    """Map typed_score to a (doc_type, confidence) tuple."""
    if typed_score >= CFG.typed_threshold:
        return ("typed", min(typed_score, CFG.max_confidence))
    elif typed_score <= CFG.handwritten_threshold:
        return ("handwritten", min(1.0 - typed_score, CFG.max_confidence))
    else:
        return ("mixed", 0.5)


def _log_classification(scores, has_signature, has_fax_header, typed_score):
    """Emit debug logs for classification diagnostics."""
    logger.debug("Classification Scores:")
    logger.debug("  stroke_score: %.3f (low=typed)", scores['stroke'])
    logger.debug("  line_score: %.3f (high=typed)", scores['line'])
    logger.debug("  angle_score: %.1f (low=typed)", scores['angle'])
    logger.debug("  edge_score: %.4f", scores['edge'])
    logger.debug("  form_score: %.3f (high=typed/form)", scores['form'])
    logger.debug("  uniformity_score: %.3f (high=typed)", scores['uniformity'])
    logger.debug("Detection Flags:")
    logger.debug("  has_signature: %s", has_signature)
    logger.debug("  has_fax_header: %s", has_fax_header)
    logger.debug("Final typed_score: %.3f", typed_score)
    logger.debug("  Thresholds: typed>=%.2f, handwritten<=%.2f, else mixed",
                 CFG.typed_threshold, CFG.handwritten_threshold)


# ============================================================================
# Feature Extractors
# ============================================================================

def _detect_signature_in_region(region):
    """Detect if a region contains a handwritten signature via connected components."""
    try:
        sig = CFG.signature
        if region.shape[0] < sig.min_height or region.shape[1] < sig.min_width:
            return False

        _, binary = cv2.threshold(region, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        total_pixels = region.shape[0] * region.shape[1]
        ink_ratio = np.count_nonzero(binary) / total_pixels if total_pixels > 0 else 0

        if ink_ratio < sig.min_ink_ratio or ink_ratio > sig.max_ink_ratio:
            return False

        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        if num_labels < 2:
            return False

        sig_components = 0
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            width = stats[i, cv2.CC_STAT_WIDTH]
            if sig.min_component_area < area < total_pixels * sig.max_component_ratio and width > sig.min_component_width:
                sig_components += 1

        return sig.min_components <= sig_components <= sig.max_components

    except cv2.error as e:
        logger.debug("_detect_signature_in_region failed: %s", e)
        return False


def _detect_fax_header(header_region):
    """Detect fax/letterhead header patterns via horizontal line morphology."""
    try:
        fax = CFG.fax_header
        if header_region.shape[0] < fax.min_height or header_region.shape[1] < fax.min_width:
            return False

        _, binary = cv2.threshold(header_region, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        kernel_w = max(fax.min_kernel_w, header_region.shape[1] // fax.kernel_divisor)
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 1))
        horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)

        total_pixels = header_region.shape[0] * header_region.shape[1]
        ink_ratio = np.count_nonzero(binary) / total_pixels if total_pixels > 0 else 0

        has_lines = np.count_nonzero(horizontal_lines) > (header_region.shape[1] * fax.line_coverage_ratio)
        has_dense_text = fax.ink_range[0] < ink_ratio < fax.ink_range[1]

        return has_lines and has_dense_text

    except cv2.error as e:
        logger.debug("_detect_fax_header failed: %s", e)
        return False


def _calculate_stroke_variance(gray):
    """
    Calculate stroke width variance using distance transform.
    Low variance = typed, high variance = handwritten.
    Returns: 0.0 (uniform/typed) to 1.0 (variable/handwritten).
    """
    try:
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

        stroke_widths = dist[dist > 0]
        if len(stroke_widths) < CFG.stroke_min_samples:
            return CFG.default_score

        mean_width = np.mean(stroke_widths)
        if mean_width < CFG.stroke_min_mean_width:
            return CFG.default_score

        cv_score = np.std(stroke_widths) / mean_width
        return min(cv_score / CFG.stroke_cv_normalizer, 1.0)

    except cv2.error as e:
        logger.debug("_calculate_stroke_variance failed: %s", e)
        return CFG.default_score


def _calculate_line_regularity(gray):
    """
    Calculate how horizontal/regular text lines are using Hough lines.
    Returns: 0.0 (irregular) to 1.0 (perfectly horizontal).
    """
    try:
        h, w = gray.shape
        edges = cv2.Canny(gray, 50, 150)

        min_line_len = max(15, w // 33)
        max_line_gap = max(5, w // 100)

        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=CFG.line_hough_threshold,
                                minLineLength=min_line_len, maxLineGap=max_line_gap)

        if lines is None or len(lines) < CFG.line_min_hough_lines:
            return CFG.default_score

        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 != 0:
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                angles.append(abs(angle))

        if not angles:
            return CFG.default_score

        tol = CFG.line_horizontal_angle_tolerance
        horizontal_count = sum(1 for a in angles if a < tol or a > (180 - tol))
        return horizontal_count / len(angles)

    except cv2.error as e:
        logger.debug("_calculate_line_regularity failed: %s", e)
        return CFG.default_score


def _calculate_angle_variance(gray):
    """
    Angle variance from ellipse fitting on contours.
    Low variance = typed, high variance = handwritten.
    """
    try:
        total_pixels = gray.shape[0] * gray.shape[1]
        min_contour_area = max(20, total_pixels // 15000)

        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        angles = []
        for contour in contours:
            if cv2.contourArea(contour) > min_contour_area and len(contour) >= 5:
                ellipse = cv2.fitEllipse(contour)
                angles.append(ellipse[2])

        if not angles:
            return CFG.default_angle_variance

        return np.var(angles)

    except cv2.error as e:
        logger.debug("_calculate_angle_variance failed: %s", e)
        return CFG.default_angle_variance


def _calculate_edge_density(gray):
    """Calculate edge pixel density ratio. Returns 0.0 to ~0.3."""
    try:
        edges = cv2.Canny(gray, 50, 150)
        total_pixels = edges.shape[0] * edges.shape[1]
        return np.count_nonzero(edges) / total_pixels if total_pixels > 0 else 0.0

    except cv2.error as e:
        logger.debug("_calculate_edge_density failed: %s", e)
        return 0.1


def _detect_form_structure(gray):
    """
    Detect horizontal/vertical lines indicating form structure.
    Returns: 0.0 (no structure) to 1.0 (strong form layout).
    """
    try:
        h, w = gray.shape
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        kernel_w = max(CFG.form_min_kernel_size, w // CFG.form_kernel_divisor)
        kernel_h = max(CFG.form_min_kernel_size, h // CFG.form_kernel_divisor)

        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 1))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_h))

        h_lines = np.count_nonzero(cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel))
        v_lines = np.count_nonzero(cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel))

        total_pixels = h * w
        line_ratio = (h_lines + v_lines) / total_pixels if total_pixels > 0 else 0
        return min(line_ratio / CFG.form_line_ratio_normalizer, 1.0)

    except cv2.error as e:
        logger.debug("_detect_form_structure failed: %s", e)
        return 0.0


def _calculate_character_uniformity(gray):
    """
    Calculate character height uniformity via connected components.
    Returns: 0.0 (variable/handwritten) to 1.0 (uniform/typed).
    """
    try:
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

        if num_labels < CFG.uniformity_min_components:
            return CFG.default_score

        heights = []
        for i in range(1, num_labels):
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]
            if 5 < h < gray.shape[0] // 3 and area > 20:
                heights.append(h)

        if len(heights) < CFG.uniformity_min_components:
            return CFG.default_score

        mean_height = np.mean(heights)
        if mean_height < 1:
            return CFG.default_score

        cv_score = np.std(heights) / mean_height
        return min(max(0, 1.0 - (cv_score / CFG.uniformity_cv_normalizer)), 1.0)

    except cv2.error as e:
        logger.debug("_calculate_character_uniformity failed: %s", e)
        return CFG.default_score
