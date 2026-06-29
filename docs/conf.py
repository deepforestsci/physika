import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "Physika"
copyright = "2026, deepforestsci"
author = "deepforestsci"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
]

mathjax3_config = {
    "tex": {
        "tags": "all",
    }
}

autodoc_mock_imports = ["torch", "numpy", "ply"]

templates_path = ["_templates"]
exclude_patterns = ["_build"]

html_theme = "sphinx_rtd_theme"
html_logo = "_static/logo.png"
html_static_path = ["_static"]
