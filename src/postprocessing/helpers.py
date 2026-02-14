"""
Shared helper functions used across postprocessing submodules.

Re-exports bbox overlap utilities from the canonical src.utils.bbox module.
"""

from src.utils.bbox import bbox_overlap_ratio_of_smaller

# Backwards-compatible alias used by signatures.py, table_validation.py, pipeline.py
bbox_overlap = bbox_overlap_ratio_of_smaller
