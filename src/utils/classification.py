"""
Document type classification (typed, handwritten, mixed).

Multi-feature analysis using stroke width, line regularity, angle variance,
edge density, form structure, character uniformity, signature isolation,
and fax/letterhead header detection.
"""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def classify_document_type(image):
    """
    Classify document as typed, handwritten, or mixed based on multiple features.

    Returns: tuple (doc_type: str, confidence: float)
        - doc_type: "typed", "handwritten", or "mixed"
        - confidence: 0.0 to 1.0

    Features used:
    1. Stroke width variance (typed = uniform, handwritten = variable)
    2. Line regularity (typed = horizontal, handwritten = slanted)
    3. Edge density patterns
    4. Contour angle variance
    5. Form structure detection (lines, boxes)
    6. Character uniformity
    7. Signature isolation (exclude bottom signature region from analysis)
    8. Fax/letterhead header detection
    """
    try:
        img_np = np.array(image)
        if len(img_np.shape) == 3:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_np

        h, w = gray.shape
        if w > 1000:
            scale = 1000 / w
            gray = cv2.resize(gray, (0, 0), fx=scale, fy=scale)
            h, w = gray.shape

        # Isolate signature region (bottom 20% of page)
        signature_region_start = int(h * 0.80)
        main_body = gray[:signature_region_start, :]
        signature_region = gray[signature_region_start:, :]

        has_signature = _detect_signature_in_region(signature_region)

        # Detect fax/letterhead header patterns in top 15% of page
        header_region = gray[:int(h * 0.15), :]
        has_fax_header = _detect_fax_header(header_region)

        # Use main body (excluding signature) for feature extraction
        analysis_region = main_body if has_signature else gray

        # Feature extraction
        stroke_score = _calculate_stroke_variance(analysis_region)
        line_score = _calculate_line_regularity(analysis_region)
        angle_score = _calculate_angle_variance(analysis_region)
        edge_score = _calculate_edge_density(analysis_region)
        form_score = _detect_form_structure(gray)  # Use full page for form detection
        uniformity_score = _calculate_character_uniformity(analysis_region)

        # Weighted voting for "typed"
        typed_score = 0.0

        # Stroke variance: low = typed
        if stroke_score < 0.3:
            typed_score += 0.20
        elif stroke_score < 0.5:
            typed_score += 0.10

        # Line regularity: high = typed
        if line_score > 0.7:
            typed_score += 0.12
        elif line_score > 0.5:
            typed_score += 0.06

        # Angle variance
        if angle_score < 400:
            typed_score += 0.10
        elif angle_score < 800:
            typed_score += 0.05
        elif angle_score > 1500 and form_score < 0.3:
            typed_score -= 0.15
        elif angle_score > 1200 and form_score < 0.3:
            typed_score -= 0.08

        # Edge density: moderate = typed
        if 0.02 < edge_score < 0.15:
            typed_score += 0.08
        elif edge_score < 0.02:
            typed_score += 0.04

        # Form structure
        if form_score > 0.5:
            typed_score += 0.30
        elif form_score > 0.2:
            typed_score += 0.20
        elif form_score > 0.1:
            typed_score += 0.12

        # Character uniformity
        if uniformity_score > 0.7:
            typed_score += 0.12
        elif uniformity_score > 0.4:
            typed_score += 0.06

        # Fax/letterhead header boost
        if has_fax_header:
            typed_score += 0.20

        # Signature on a form
        if has_signature and form_score > 0.2:
            typed_score += 0.10

        # Ruled paper handwriting detection
        is_ruled_paper_handwriting = (
            line_score >= 0.95 and
            form_score < 0.25 and
            300 < angle_score < 700
        )
        if is_ruled_paper_handwriting:
            typed_score -= 0.12
            logger.debug("Ruled paper handwriting detected: -0.12 adjustment")

        # Sparse/blank form detection
        is_sparse_form = (
            edge_score < 0.02 and
            form_score > 0.15 and
            not is_ruled_paper_handwriting
        )
        if is_sparse_form and typed_score < 0.65:
            typed_score = 0.65
            logger.debug("Sparse/blank form detected: forcing typed classification")

        # Clamp to valid range
        typed_score = max(0.0, min(typed_score, 1.0))

        # Debug logging
        logger.debug("Classification Scores:")
        logger.debug("  stroke_score: %.3f (low=typed)", stroke_score)
        logger.debug("  line_score: %.3f (high=typed)", line_score)
        logger.debug("  angle_score: %.1f (low=typed)", angle_score)
        logger.debug("  edge_score: %.4f", edge_score)
        logger.debug("  form_score: %.3f (high=typed/form)", form_score)
        logger.debug("  uniformity_score: %.3f (high=typed)", uniformity_score)
        logger.debug("Detection Flags:")
        logger.debug("  has_signature: %s", has_signature)
        logger.debug("  has_fax_header: %s", has_fax_header)
        logger.debug("Final typed_score: %.3f", typed_score)
        logger.debug("  Thresholds: typed>=0.65, handwritten<=0.45, else mixed")

        # Determine classification
        if typed_score >= 0.65:
            return ("typed", min(typed_score, 0.95))
        elif typed_score <= 0.45:
            return ("handwritten", min(1.0 - typed_score, 0.95))
        else:
            return ("mixed", 0.5)

    except Exception as e:
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
        if hasattr(image, 'mode'):  # PIL Image
            gray = np.array(image.convert('L'))
        else:
            gray = image

        h, w = gray.shape[:2] if len(gray.shape) >= 2 else (gray.shape[0], 1)

        signature_region_start = int(h * 0.80)
        signature_region = gray[signature_region_start:, :]

        has_signature = _detect_signature_in_region(signature_region)

        if has_signature:
            return {
                'has_signature': True,
                'region_bbox': [0.0, 0.80, 1.0, 1.0]
            }
        else:
            return {
                'has_signature': False,
                'region_bbox': None
            }
    except Exception:
        return {
            'has_signature': False,
            'region_bbox': None
        }


def _detect_signature_in_region(region):
    """
    Detect if a region (typically bottom 20% of page) contains a handwritten signature.
    """
    try:
        if region.shape[0] < 20 or region.shape[1] < 50:
            return False

        _, binary = cv2.threshold(region, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        ink_pixels = np.count_nonzero(binary)
        total_pixels = region.shape[0] * region.shape[1]
        ink_ratio = ink_pixels / total_pixels if total_pixels > 0 else 0

        if ink_ratio < 0.003 or ink_ratio > 0.15:
            return False

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

        if num_labels < 2:
            return False

        sig_components = 0
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            width = stats[i, cv2.CC_STAT_WIDTH]

            if 50 < area < total_pixels * 0.3 and width > 20:
                sig_components += 1

        return 1 <= sig_components <= 15

    except Exception:
        return False


def _detect_fax_header(header_region):
    """
    Detect fax/letterhead header patterns.
    """
    try:
        if header_region.shape[0] < 30 or header_region.shape[1] < 100:
            return False

        _, binary = cv2.threshold(header_region, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        kernel_w = max(25, header_region.shape[1] // 20)

        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 1))
        horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
        h_line_pixels = np.count_nonzero(horizontal_lines)

        ink_pixels = np.count_nonzero(binary)
        total_pixels = header_region.shape[0] * header_region.shape[1]
        ink_ratio = ink_pixels / total_pixels if total_pixels > 0 else 0

        has_lines = h_line_pixels > (header_region.shape[1] * 0.1)
        has_dense_text = 0.02 < ink_ratio < 0.25

        return has_lines and has_dense_text

    except Exception:
        return False


def _calculate_stroke_variance(gray):
    """
    Calculate stroke width variance using distance transform.
    Low variance = typed, high variance = handwritten.
    """
    try:
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

        stroke_widths = dist[dist > 0]

        if len(stroke_widths) < 100:
            return 0.5

        mean_width = np.mean(stroke_widths)
        if mean_width < 0.1:
            return 0.5

        cv_score = np.std(stroke_widths) / mean_width
        return min(cv_score / 0.5, 1.0)

    except Exception:
        return 0.5


def _calculate_line_regularity(gray):
    """
    Calculate how horizontal/regular text lines are.
    """
    try:
        h, w = gray.shape

        edges = cv2.Canny(gray, 50, 150)

        min_line_len = max(15, w // 33)
        max_line_gap = max(5, w // 100)

        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50,
                                minLineLength=min_line_len, maxLineGap=max_line_gap)

        if lines is None or len(lines) < 5:
            return 0.5

        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 != 0:
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                angles.append(abs(angle))

        if not angles:
            return 0.5

        horizontal_count = sum(1 for a in angles if a < 5 or a > 175)
        regularity = horizontal_count / len(angles)

        return regularity

    except Exception:
        return 0.5


def _calculate_angle_variance(gray):
    """
    Angle variance calculation from contour fitting.
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
            return 1000

        return np.var(angles)

    except Exception:
        return 1000


def _calculate_edge_density(gray):
    """
    Calculate edge density ratio.
    """
    try:
        edges = cv2.Canny(gray, 50, 150)
        edge_pixels = np.count_nonzero(edges)
        total_pixels = edges.shape[0] * edges.shape[1]

        return edge_pixels / total_pixels if total_pixels > 0 else 0.0

    except Exception:
        return 0.1


def _detect_form_structure(gray):
    """
    Detect horizontal/vertical lines indicating form structure.
    """
    try:
        h, w = gray.shape

        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        kernel_w = max(20, w // 25)
        kernel_h = max(20, h // 25)

        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 1))
        horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)

        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_h))
        vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)

        h_line_pixels = np.count_nonzero(horizontal_lines)
        v_line_pixels = np.count_nonzero(vertical_lines)
        total_line_pixels = h_line_pixels + v_line_pixels

        total_pixels = h * w

        line_ratio = total_line_pixels / total_pixels if total_pixels > 0 else 0
        form_score = min(line_ratio / 0.01, 1.0)

        return form_score

    except Exception:
        return 0.0


def _calculate_character_uniformity(gray):
    """
    Calculate character height uniformity.
    """
    try:
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

        if num_labels < 10:
            return 0.5

        heights = []
        for i in range(1, num_labels):
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]
            if 5 < h < gray.shape[0] // 3 and area > 20:
                heights.append(h)

        if len(heights) < 10:
            return 0.5

        mean_height = np.mean(heights)
        std_height = np.std(heights)

        if mean_height < 1:
            return 0.5

        cv_score = std_height / mean_height
        uniformity = max(0, 1.0 - (cv_score / 0.4))

        return min(uniformity, 1.0)

    except Exception:
        return 0.5
