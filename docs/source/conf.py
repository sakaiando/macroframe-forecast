# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join('..', '..')))

project = "mff"
copyright = "2024, IMF"
author = "Ando Sakai, Doga Bilgin and Sultan Orazbayev"
release = "0.1"


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
extensions = [
    "sphinx.ext.autodoc",
    "numpydoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.intersphinx",
    "sphinx.ext.doctest",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "**.ipynb_checkpoints", "**.ipynb"]

# generate autosummary even if no references
autosummary_generate = True

# Members and inherited-members default to showing methods and attributes from a
# class or those inherited.
# Member-order orders the documentation in the order of how the members are defined in
# the source code.
autodoc_default_options = {
    "members": True,
    "inherited-members": True,
    "member-order": "bysource",
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = []
