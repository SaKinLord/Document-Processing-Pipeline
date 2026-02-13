"""
Image loading, conversion, cropping, denoising, and deskewing utilities.
"""

import logging

import cv2
import numpy as np
from PIL import Image
import pypdfium2 as pdfium

logger = logging.getLogger(__name__)


def load_image(image_path):
    """Loads an image from a file path."""
    try:
        image = Image.open(image_path).convert("RGB")
        return image
    except Exception as e:
        logger.error("Error loading image %s: %s", image_path, e)
        return None


def convert_pdf_to_images(pdf_path, dpi=200):
    """Converts a PDF file to a list of PIL Images."""
    images = []
    try:
        pdf = pdfium.PdfDocument(pdf_path)
        for i in range(len(pdf)):
            page = pdf[i]
            bitmap = page.render(scale=dpi/72)
            pil_image = bitmap.to_pil()
            images.append(pil_image.convert("RGB"))
    except Exception as e:
        logger.error("Error converting PDF %s: %s", pdf_path, e)
    return images


def crop_image(image, bbox):
    """Crops an image given a bounding box [x1, y1, x2, y2]."""
    return image.crop(bbox)


def pad_bbox(bbox, padding, image_width, image_height):
    """
    Pad a bounding box by a given number of pixels, ensuring it stays within image bounds.

    This is particularly important for handwritten text where Surya's tight cropping
    may cut off descenders (g, y, p, q) which TrOCR needs to see for accurate recognition.

    Args:
        bbox: Bounding box [x1, y1, x2, y2]
        padding: Number of pixels to add on each side
        image_width: Width of the source image
        image_height: Height of the source image

    Returns:
        Padded bounding box [x1, y1, x2, y2] clamped to image bounds
    """
    x1, y1, x2, y2 = bbox
    return [
        max(0, x1 - padding),
        max(0, y1 - padding),
        min(image_width, x2 + padding),
        min(image_height, y2 + padding)
    ]


def estimate_noise(image_np):
    """
    Estimates noise level using a simple median blur difference method.
    Higher score means more noise.
    """
    try:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

        h, w = gray.shape
        if w > 1000:
            scale = 1000 / w
            gray = cv2.resize(gray, (0, 0), fx=scale, fy=scale)

        median = cv2.medianBlur(gray, 3)
        diff = cv2.absdiff(gray, median)
        score = np.mean(diff)
        return score
    except Exception:
        return 999.0


def denoise_image(image):
    """
    Applies simple denoising to an image using OpenCV if noise is detected.
    Intended for scanned documents with salt-and-pepper noise.
    """
    try:
        img_np = np.array(image)

        noise_score = estimate_noise(img_np)
        if noise_score < 0.8:
            return image

        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        denoised_bgr = cv2.fastNlMeansDenoisingColored(img_bgr, None, 10, 10, 7, 21)
        denoised_rgb = cv2.cvtColor(denoised_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(denoised_rgb)
    except Exception as e:
        logger.error("Error denoising image: %s", e)
        return image


def deskew_image(image, angle_threshold=0.5, max_angle=15.0):
    """
    Detect and correct document skew using Hough line detection.

    Analyzes horizontal lines in the document to estimate rotation angle,
    then applies correction if the skew exceeds the threshold.

    Args:
        image: PIL Image to deskew
        angle_threshold: Minimum skew angle (degrees) to trigger correction
        max_angle: Maximum correction angle (ignore if skew seems too extreme)

    Returns:
        Deskewed PIL Image (or original if no significant skew detected)
    """
    try:
        img_np = np.array(image.convert('L'))
        h, w = img_np.shape

        edges = cv2.Canny(img_np, 50, 150, apertureSize=3)

        max_line_gap = max(5, w // 100)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50,
                                minLineLength=w//4, maxLineGap=max_line_gap)

        if lines is None or len(lines) < 10:
            return image

        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(x2 - x1) > 10:
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                if -30 < angle < 30:
                    angles.append(angle)

        if len(angles) < 5:
            return image

        median_angle = np.median(angles)

        if abs(median_angle) < angle_threshold or abs(median_angle) > max_angle:
            return image

        logger.debug("Detected skew: %.2f degrees, applying correction", median_angle)

        img_color = np.array(image)
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)

        cos = np.abs(rotation_matrix[0, 0])
        sin = np.abs(rotation_matrix[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))

        rotation_matrix[0, 2] += (new_w / 2) - center[0]
        rotation_matrix[1, 2] += (new_h / 2) - center[1]

        rotated = cv2.warpAffine(img_color, rotation_matrix, (new_w, new_h),
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(255, 255, 255))

        return Image.fromarray(rotated)

    except Exception as e:
        logger.warning("Deskew failed: %s", e)
        return image
