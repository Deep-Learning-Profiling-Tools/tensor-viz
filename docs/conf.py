"""Sphinx configuration for tensor-viz."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON_SRC = ROOT / "python" / "src"
TYPEDOC_DIR = ROOT / "docs" / "_extra"

sys.path.insert(0, str(PYTHON_SRC))

project = "tensor-viz"
author = "tensor-viz"
root_doc = "index"
extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]
templates_path: list[str] = []
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"
myst_heading_anchors = 3
html_theme = "sphinx_rtd_theme"
html_static_path: list[str] = []
html_extra_path = ["_extra"] if TYPEDOC_DIR.exists() else []
