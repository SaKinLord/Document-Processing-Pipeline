"""
Utilities package for the Document Processing Pipeline.

Re-exports all public functions for backwards-compatible imports.
"""

from .image import (
    load_image,
    convert_pdf_to_images,
    crop_image,
    pad_bbox,
    estimate_noise,
    denoise_image,
    deskew_image,
)
from .bbox import (
    split_line_bbox_to_words,
    is_bbox_too_large,
)
from .classification import (
    classify_document_type,
    detect_signature_region,
)
from .text import (
    SpellCorrector,
    cluster_text_rows,
)

__all__ = [
    # Image
    'load_image',
    'convert_pdf_to_images',
    'crop_image',
    'pad_bbox',
    'estimate_noise',
    'denoise_image',
    'deskew_image',
    # Bbox
    'split_line_bbox_to_words',
    'is_bbox_too_large',
    # Classification
    'classify_document_type',
    'detect_signature_region',
    # Text
    'SpellCorrector',
    'cluster_text_rows',
]
