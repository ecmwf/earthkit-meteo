# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

on_rtd = os.environ.get("READTHEDOCS") == "True"

if on_rtd:
    version = os.environ.get("READTHEDOCS_VERSION", "latest")
    release = version
else:
    version = "dev"
    release = "dev"

rtd_version = version if version != "latest" else "develop"
rtd_version_type = os.environ.get("READTHEDOCS_VERSION_TYPE", "branch")

if rtd_version_type in ("branch", "tag"):
    source_branch = rtd_version
else:
    source_branch = "main"

sys.path.insert(0, os.path.abspath("../../src"))
sys.path.insert(0, os.path.abspath("./"))
sys.path.insert(0, os.path.abspath("../"))
sys.path.append(os.path.abspath("./_ext"))

project = "earthkit-meteo"
copyright = "2025, European Centre for Medium-Range Weather Forecasts (ECMWF)"
author = "European Centre for Medium-Range Weather Forecasts (ECMWF)"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    # Automatically extracts documentation from your Python docstrings
    "sphinx.ext.autodoc",
    # Supports Google-style and NumPy-style docstrings
    "sphinx.ext.napoleon",
    # Renders LaTeX math in HTML using MathJax
    "sphinx.ext.mathjax",
    # Option to click viewcode
    "sphinx.ext.viewcode",
    # Links to the documentation of other projects via cross-references
    # "sphinx.ext.intersphinx",
    # Generates summary tables for modules/classes/functions
    # "sphinx.ext.autosummary",
    # Allows citing BibTeX bibliographic entries in reStructuredText
    # "sphinxcontrib.bibtex",
    # Tests snippets in documentation by running embedded Python examples
    # "sphinx.ext.doctest",
    # Checks documentation coverage of the codebase
    # "sphinx.ext.coverage",
    # Adds .nojekyll file and helps configure docs for GitHub Pages hosting
    # "sphinx.ext.githubpages",
    # Adds "Edit on GitHub" links to documentation pages
    # "edit_on_github",
    # Adds "Edit on GitHub" links to documentation pages
    # "sphinx_github_style",
    # Option to link to code
    # "sphinx.ext.linkcode",
    # Automatically includes type hints from function signatures into the documentation
    # "sphinx_autodoc_typehints",
    # Integrates Jupyter Notebooks into Sphinx
    "nbsphinx",
    # Simplifies linking to external resources with short aliases
    # "sphinx.ext.extlinks"
    "autoapi.extension",
    "sphinx_issues",
    "sphinx_copybutton",
    "xref",
]

# autodoc configuration
autodoc_typehints = "description"
# autodoc_typehints = "signature"

# autoapi configuration
autoapi_dirs = ["../src/earthkit"]
autoapi_ignore = ["*/_version.py", "sphinxext/*"]
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "imported-members",
    "inherited-members",
]
autoapi_root = "autoapi"
autoapi_member_order = "alphabetical"
autoapi_add_toctree_entry = False
autoapi_own_page_level = "function"
autoapi_python_use_implicit_namespaces = True

# napoleon configuration
napoleon_google_docstring = False  #
napoleon_numpy_docstring = True
napoleon_preprocess_types = True

# Path to GitHub repo {group}/{project}  (note that `group` is the GitHub user or organization)
issues_github_path = "ecmwf/earthkit-meteo"

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The suffix of source filenames.
source_suffix = ".rst"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "furo"

html_static_path = ["_static"]

html_css_files = [
    "custom.css",
]

xref_links = {
    "earthkit": ("earthkit", "https://earthkit.readthedocs.io/en/latest/"),
    "earthkit-data": (
        "earthkit-data",
        "https://earthkit-data.readthedocs.io/en/latest/",
    ),
    "earthkit-plots": (
        "earthkit-plots",
        "https://earthkit-plots.readthedocs.io/en/latest/",
    ),
    "earthkit-utils": (
        "earthkit-utils",
        "https://github.com/ecmwf/earthkit-utils",
    ),
    "fieldlist": (
        "fieldlist",
        "https://earthkit-data.readthedocs.io/en/latest/guide/data_format/grib.html",
    ),
    "field": (
        "field",
        "https://earthkit-data.readthedocs.io/en/latest/guide/data_format/grib.html",
    ),
}

html_theme_options = {
    "light_css_variables": {
        "color-sidebar-background": "#131320",
        # "color-background-primary": "", # leave as default to avoid overriding the light theme background
        "color-sidebar-link-text": "#ffffff",
        "color-sidebar-brand-text": "#ffffff",
        "color-sidebar-caption-text": "#ffffff",
        "color-brand-primary": "#FCE54B",
        "color-brand-content": "#5f8dd3",
        "color-sidebar-item-background--hover": "#001F3F",
        "color-sidebar-item-expander-background--hover": "#001F3F",
    },
    "dark_css_variables": {
        "color-sidebar-background": "#131320",
        "color-background-primary": "#131320",
        "color-sidebar-link-text": "#ffffff",
        "color-sidebar-brand-text": "#ffffff",
        "color-sidebar-caption-text": "#ffffff",
        "color-brand-primary": "#FCE54B",
        "color-brand-content": "#5f8dd3",
        "color-sidebar-item-background--hover": "#001F3F",
        "color-sidebar-item-expander-background--hover": "#001F3F",
    },
    "light_logo": "earthkit-meteo-dark.svg",
    "dark_logo": "earthkit-meteo-dark.svg",
    "source_repository": "https://github.com/ecmwf/earthkit-meteo/",
    "source_branch": source_branch,
    "source_directory": "docs/source",
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/ecmwf/earthkit-meteo",
            "html": """
                <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">
                    <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path>
                </svg>
            """,
            "class": "",
        },
    ],
}


def setup(app):
    from skip_api_rules import _skip_api_items

    app.connect("autoapi-skip-member", _skip_api_items)
