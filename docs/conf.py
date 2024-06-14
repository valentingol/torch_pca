"""Configuration file for the Sphinx documentation builder."""

# pylint: disable=all
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import sys

from setuptools_scm import get_version

sys.path.insert(0, os.path.abspath("../"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Pytorch PCA"
copyright = "2024, Valentin Goldite"  # noqa A001
author = "Valentin Goldite"
try:
    release = get_version()
except:  # noqa E722
    release = get_version(root="..", relative_to=__file__)

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.inheritance_diagram",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
]

master_doc = "index"
autoapi_type = "python"
autoapi_dirs = ["src/torch_pca"]

autodoc_default_options = {
    "member-order": "bysource",
    "undoc-members": True,
}

add_module_names = False
autoclass_content = "both"
napoleon_use_param = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/", None),
    "numpy": ("http://docs.scipy.org/doc/numpy/", None),
}

templates_path = ["_templates"]
exclude_patterns = ["_build"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

html_theme_options = {
    "canonical_url": "",
    "analytics_id": "UA-XXXXXXX-1",
    "logo_only": False,
    "display_version": True,
    "prev_next_buttons_location": "both",
    "style_external_links": "#ff9900",
    "style_nav_header_background": "#ff9900",
    # Toc options
    "collapse_navigation": False,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
}
html_context = {
    "display_github": True,  # Integrate GitHub
    "github_user": "valentingol",  # Username
    "github_repo": "torch_cpa",  # Repo name
    "github_version": "main",  # Version
    "conf_py_path": "/docs/",  # Path in the checkout to the docs root
}
