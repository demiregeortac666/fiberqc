"""Sphinx configuration for the fiberqc documentation."""

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

# -- Project ----------------------------------------------------------------
project = "fiberqc"
author = "Demir Ege Ortaç"
copyright = "2026, Demir Ege Ortaç"
release = "0.1.0"

# -- Extensions -------------------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "numpydoc",
]

autosummary_generate = True          # build stub pages for every listed object
numpydoc_show_class_members = False  # let autosummary handle members (no duplicates)
autodoc_typehints = "none"
add_module_names = False             # show `multiverse`, not `fiberqc.multiverse`
autodoc_member_order = "bysource"

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- HTML / theme -----------------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_title = "fiberqc"
html_static_path = ["_static"]
html_logo = "_static/logo.svg"
html_favicon = "_static/logo-mark.svg"

html_theme_options = {
    "github_url": "https://github.com/demiregeortac666/fiberqc",
    "show_toc_level": 2,
    "navigation_with_keys": True,
    "icon_links": [],
}

# -- Intersphinx ------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
}
