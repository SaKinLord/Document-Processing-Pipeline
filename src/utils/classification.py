"""
Document type classification (typed, handwritten, mixed).

Multi-feature analysis using stroke width, line regularity, angle variance,
edge density, form structure, character uniformity, signature isolation,
and fax/letterhead header detection.
"""

import logging

import cv2
import numpy as np

from src.config import CONFIG

logger = logging.getLogger(__name__)

# ============================================================================
# Configuration Constants
# ============================================================================

# Image preprocessing
MAX_ANALYSIS_WIDTH = 1000       # Resize images wider than this for analysis

# Page region ratios (signature ratio imported from centralized config)
SIGNATURE_REGION_RATIO = CONFIG.signature_region_ratio
HEADER_REGION_RATIO = 0.15      # Top 15% of page = fax/letterhead region

# Minimum region dimensions (pixels) for analysis
MIN_SIGNATURE_HEIGHT = 20
MIN_SIGNATURE_WIDTH = 50
MIN_HEADER_HEIGHT = 30
MIN_HEADER_WIDTH = 100

# Signature detection
MIN_INK_RATIO = 0.003           # Too little ink = no signature
MAX_INK_RATIO = 0.15            # Too much ink = printed text, not signature
MIN_SIG_COMPONENT_AREA = 50     # Minimum connected component area
MAX_SIG_COMPONENT_RATIO = 0.3   # Maximum component area as ratio of region
MIN_SIG_COMPONENT_WIDTH = 20    # Minimum component width
MIN_SIG_COMPONENTS = 1
MAX_SIG_COMPONENTS = 15

# Fax header detection
MIN_FAX_HEADER_KERNEL_W = 25
FAX_HEADER_KERNEL_DIVISOR = 20
FAX_LINE_COVERAGE_RATIO = 0.1   # Horizontal lines must span 10% of width
FAX_INK_RANGE = (0.02, 0.25)    # Valid ink density range for fax headers

# Stroke variance thresholds
MIN_STROKE_SAMPLES = 100        # Minimum stroke width samples needed
MIN_MEAN_STROKE_WIDTH = 0.1     # Below this, variance is meaningless
STROKE_CV_NORMALIZER = 0.5      # Coefficient of variation normalization factor

# Stroke score → typed_score contribution
STROKE_LOW_THRESH = 0.3         # Below = strong typed signal
STROKE_MED_THRESH = 0.5         # Below = moderate typed signal
STROKE_LOW_WEIGHT = 0.20
STROKE_MED_WEIGHT = 0.10

# Line regularity thresholds
MIN_HOUGH_LINES = 5             # Minimum lines for regularity analysis
HOUGH_THRESHOLD = 50
HORIZONTAL_ANGLE_TOLERANCE = 5  # Degrees from horizontal to count as "regular"

# Line score → typed_score contribution
LINE_HIGH_THRESH = 0.7          # Above = strong typed signal
LINE_MED_THRESH = 0.5           # Above = moderate typed signal
LINE_HIGH_WEIGHT = 0.12
LINE_MED_WEIGHT = 0.06

# Angle variance thresholds (from contour fitting)
ANGLE_LOW_THRESH = 400          # Below = typed
ANGLE_MED_THRESH = 800          # Below = moderate typed
ANGLE_HIGH_THRESH = 1500        # Above (with low form) = handwritten
ANGLE_MED_HIGH_THRESH = 1200    # Above (with low form) = moderate handwritten
ANGLE_FORM_GATE = 0.3           # Form score must be below this for handwriting penalty
ANGLE_LOW_WEIGHT = 0.10
ANGLE_MED_WEIGHT = 0.05
ANGLE_HIGH_PENALTY = -0.15
ANGLE_MED_HIGH_PENALTY = -0.08

# Edge density thresholds
EDGE_TYPED_RANGE = (0.02, 0.15) # Edge density range indicating typed text
EDGE_SPARSE_THRESH = 0.02       # Below = sparse (still slightly typed)
EDGE_TYPED_WEIGHT = 0.08
EDGE_SPARSE_WEIGHT = 0.04

# Form structure thresholds
FORM_HIGH_THRESH = 0.5          # Strong form structure
FORM_MED_THRESH = 0.2           # Moderate form structure
FORM_LOW_THRESH = 0.1           # Weak form structure
FORM_HIGH_WEIGHT = 0.30
FORM_MED_WEIGHT = 0.20
FORM_LOW_WEIGHT = 0.12
FORM_LINE_RATIO_NORMALIZER = 0.01  # Normalize line pixel ratio to 0-1 score
MIN_FORM_KERNEL_SIZE = 20
FORM_KERNEL_DIVISOR = 25

# Character uniformity thresholds
MIN_UNIFORM_COMPONENTS = 10     # Minimum components for meaningful analysis
UNIFORMITY_HIGH_THRESH = 0.7
UNIFORMITY_MED_THRESH = 0.4
UNIFORMITY_HIGH_WEIGHT = 0.12
UNIFORMITY_MED_WEIGHT = 0.06
UNIFORMITY_CV_NORMALIZER = 0.4  # CV normalization factor

# Fax header boost
FAX_HEADER_WEIGHT = 0.20

# Signature on form boost
SIG_ON_FORM_THRESH = 0.2        # Minimum form score for signature boost
SIG_ON_FORM_WEIGHT = 0.10

# Ruled paper handwriting detection
RULED_PAPER_LINE_THRESH = 0.95
RULED_PAPER_FORM_THRESH = 0.25
RULED_PAPER_ANGLE_RANGE = (300, 700)
RULED_PAPER_PENALTY = -0.12

# Sparse/blank form detection
SPARSE_FORM_EDGE_THRESH = 0.02
SPARSE_FORM_FORM_THRESH = 0.15
SPARSE_FORM_FLOOR = 0.65       # Force typed_score to this minimum

# Final classification boundaries
TYPED_THRESHOLD = 0.65          # typed_score >= this → "typed"
HANDWRITTEN_THRESHOLD = 0.45    # typed_score <= this → "handwritten"
MAX_CLASSIFICATION_CONFIDENCE = 0.95

# Feature default (returned on analysis failure)
DEFAULT_SCORE = 0.5
DEFAULT_ANGLE_VARIANCE = 1000


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
        sig_start = int(h * SIGNATURE_REGION_RATIO)
        main_body = gray[:sig_start, :]
        signature_region = gray[sig_start:, :]
        header_region = gray[:int(h * HEADER_REGION_RATIO), :]

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

        sig_start = int(h * SIGNATURE_REGION_RATIO)
        signature_region = gray[sig_start:, :]

        if _detect_signature_in_region(signature_region):
            return {
                'has_signature': True,
                'region_bbox': [0.0, SIGNATURE_REGION_RATIO, 1.0, 1.0]
            }
        return {'has_signature': False, 'region_bbox': None}

    except (cv2.error, ValueError, IndexError) as e:
        logger.warning("detect_signature_region failed: %s", e)
        return {'has_signature': False, 'region_bbox': None}


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
    if w > MAX_ANALYSIS_WIDTH:
        scale = MAX_ANALYSIS_WIDTH / w
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
    if scores['stroke'] < STROKE_LOW_THRESH:
        typed_score += STROKE_LOW_WEIGHT
    elif scores['stroke'] < STROKE_MED_THRESH:
        typed_score += STROKE_MED_WEIGHT

    # Line regularity: high = typed
    if scores['line'] > LINE_HIGH_THRESH:
        typed_score += LINE_HIGH_WEIGHT
    elif scores['line'] > LINE_MED_THRESH:
        typed_score += LINE_MED_WEIGHT

    # Angle variance
    if scores['angle'] < ANGLE_LOW_THRESH:
        typed_score += ANGLE_LOW_WEIGHT
    elif scores['angle'] < ANGLE_MED_THRESH:
        typed_score += ANGLE_MED_WEIGHT
    elif scores['angle'] > ANGLE_HIGH_THRESH and scores['form'] < ANGLE_FORM_GATE:
        typed_score += ANGLE_HIGH_PENALTY
    elif scores['angle'] > ANGLE_MED_HIGH_THRESH and scores['form'] < ANGLE_FORM_GATE:
        typed_score += ANGLE_MED_HIGH_PENALTY

    # Edge density: moderate = typed
    if EDGE_TYPED_RANGE[0] < scores['edge'] < EDGE_TYPED_RANGE[1]:
        typed_score += EDGE_TYPED_WEIGHT
    elif scores['edge'] < EDGE_SPARSE_THRESH:
        typed_score += EDGE_SPARSE_WEIGHT

    # Form structure (strongest signal)
    if scores['form'] > FORM_HIGH_THRESH:
        typed_score += FORM_HIGH_WEIGHT
    elif scores['form'] > FORM_MED_THRESH:
        typed_score += FORM_MED_WEIGHT
    elif scores['form'] > FORM_LOW_THRESH:
        typed_score += FORM_LOW_WEIGHT

    # Character uniformity
    if scores['uniformity'] > UNIFORMITY_HIGH_THRESH:
        typed_score += UNIFORMITY_HIGH_WEIGHT
    elif scores['uniformity'] > UNIFORMITY_MED_THRESH:
        typed_score += UNIFORMITY_MED_WEIGHT

    # Fax/letterhead header boost
    if has_fax_header:
        typed_score += FAX_HEADER_WEIGHT

    # Signature on a form
    if has_signature and scores['form'] > SIG_ON_FORM_THRESH:
        typed_score += SIG_ON_FORM_WEIGHT

    # Ruled paper handwriting penalty
    is_ruled = (
        scores['line'] >= RULED_PAPER_LINE_THRESH
        and scores['form'] < RULED_PAPER_FORM_THRESH
        and RULED_PAPER_ANGLE_RANGE[0] < scores['angle'] < RULED_PAPER_ANGLE_RANGE[1]
    )
    if is_ruled:
        typed_score += RULED_PAPER_PENALTY
        logger.debug("Ruled paper handwriting detected: %.2f adjustment", RULED_PAPER_PENALTY)

    # Sparse/blank form floor
    is_sparse = (
        scores['edge'] < SPARSE_FORM_EDGE_THRESH
        and scores['form'] > SPARSE_FORM_FORM_THRESH
        and not is_ruled
    )
    if is_sparse and typed_score < SPARSE_FORM_FLOOR:
        typed_score = SPARSE_FORM_FLOOR
        logger.debug("Sparse/blank form detected: forcing typed classification")

    return max(0.0, min(typed_score, 1.0))


def _classify_from_score(typed_score):
    """Map typed_score to a (doc_type, confidence) tuple."""
    if typed_score >= TYPED_THRESHOLD:
        return ("typed", min(typed_score, MAX_CLASSIFICATION_CONFIDENCE))
    elif typed_score <= HANDWRITTEN_THRESHOLD:
        return ("handwritten", min(1.0 - typed_score, MAX_CLASSIFICATION_CONFIDENCE))
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
                 TYPED_THRESHOLD, HANDWRITTEN_THRESHOLD)


# ============================================================================
# Feature Extractors
# ============================================================================

def _detect_signature_in_region(region):
    """Detect if a region contains a handwritten signature via connected components."""
    try:
        if region.shape[0] < MIN_SIGNATURE_HEIGHT or region.shape[1] < MIN_SIGNATURE_WIDTH:
            return False

        _, binary = cv2.threshold(region, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        total_pixels = region.shape[0] * region.shape[1]
        ink_ratio = np.count_nonzero(binary) / total_pixels if total_pixels > 0 else 0

        if ink_ratio < MIN_INK_RATIO or ink_ratio > MAX_INK_RATIO:
            return False

        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        if num_labels < 2:
            return False

        sig_components = 0
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            width = stats[i, cv2.CC_STAT_WIDTH]
            if MIN_SIG_COMPONENT_AREA < area < total_pixels * MAX_SIG_COMPONENT_RATIO and width > MIN_SIG_COMPONENT_WIDTH:
                sig_components += 1

        return MIN_SIG_COMPONENTS <= sig_components <= MAX_SIG_COMPONENTS

    except cv2.error as e:
        logger.debug("_detect_signature_in_region failed: %s", e)
        return False


def _detect_fax_header(header_region):
    """Detect fax/letterhead header patterns via horizontal line morphology."""
    try:
        if header_region.shape[0] < MIN_HEADER_HEIGHT or header_region.shape[1] < MIN_HEADER_WIDTH:
            return False

        _, binary = cv2.threshold(header_region, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        kernel_w = max(MIN_FAX_HEADER_KERNEL_W, header_region.shape[1] // FAX_HEADER_KERNEL_DIVISOR)
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 1))
        horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)

        total_pixels = header_region.shape[0] * header_region.shape[1]
        ink_ratio = np.count_nonzero(binary) / total_pixels if total_pixels > 0 else 0

        has_lines = np.count_nonzero(horizontal_lines) > (header_region.shape[1] * FAX_LINE_COVERAGE_RATIO)
        has_dense_text = FAX_INK_RANGE[0] < ink_ratio < FAX_INK_RANGE[1]

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
        if len(stroke_widths) < MIN_STROKE_SAMPLES:
            return DEFAULT_SCORE

        mean_width = np.mean(stroke_widths)
        if mean_width < MIN_MEAN_STROKE_WIDTH:
            return DEFAULT_SCORE

        cv_score = np.std(stroke_widths) / mean_width
        return min(cv_score / STROKE_CV_NORMALIZER, 1.0)

    except cv2.error as e:
        logger.debug("_calculate_stroke_variance failed: %s", e)
        return DEFAULT_SCORE


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

        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=HOUGH_THRESHOLD,
                                minLineLength=min_line_len, maxLineGap=max_line_gap)

        if lines is None or len(lines) < MIN_HOUGH_LINES:
            return DEFAULT_SCORE

        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 != 0:
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                angles.append(abs(angle))

        if not angles:
            return DEFAULT_SCORE

        horizontal_count = sum(1 for a in angles if a < HORIZONTAL_ANGLE_TOLERANCE or a > (180 - HORIZONTAL_ANGLE_TOLERANCE))
        return horizontal_count / len(angles)

    except cv2.error as e:
        logger.debug("_calculate_line_regularity failed: %s", e)
        return DEFAULT_SCORE


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
            return DEFAULT_ANGLE_VARIANCE

        return np.var(angles)

    except cv2.error as e:
        logger.debug("_calculate_angle_variance failed: %s", e)
        return DEFAULT_ANGLE_VARIANCE


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

        kernel_w = max(MIN_FORM_KERNEL_SIZE, w // FORM_KERNEL_DIVISOR)
        kernel_h = max(MIN_FORM_KERNEL_SIZE, h // FORM_KERNEL_DIVISOR)

        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 1))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_h))

        h_lines = np.count_nonzero(cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel))
        v_lines = np.count_nonzero(cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel))

        total_pixels = h * w
        line_ratio = (h_lines + v_lines) / total_pixels if total_pixels > 0 else 0
        return min(line_ratio / FORM_LINE_RATIO_NORMALIZER, 1.0)

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

        if num_labels < MIN_UNIFORM_COMPONENTS:
            return DEFAULT_SCORE

        heights = []
        for i in range(1, num_labels):
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]
            if 5 < h < gray.shape[0] // 3 and area > 20:
                heights.append(h)

        if len(heights) < MIN_UNIFORM_COMPONENTS:
            return DEFAULT_SCORE

        mean_height = np.mean(heights)
        if mean_height < 1:
            return DEFAULT_SCORE

        cv_score = np.std(heights) / mean_height
        return min(max(0, 1.0 - (cv_score / UNIFORMITY_CV_NORMALIZER)), 1.0)

    except cv2.error as e:
        logger.debug("_calculate_character_uniformity failed: %s", e)
        return DEFAULT_SCORE
