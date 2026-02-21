"""
Pytest configuration for the test suite.

Adds the project root to sys.path so that imports like
``from src.postprocessing import ...`` and ``from flexible_evaluation import ...``
work without per-file sys.path hacks.
"""

import sys
from pathlib import Path

# Add project root so ``from src.*`` imports work
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Add tests/ directory so ``from flexible_evaluation import ...`` works
_tests_dir = str(Path(__file__).resolve().parent)
if _tests_dir not in sys.path:
    sys.path.insert(0, _tests_dir)
