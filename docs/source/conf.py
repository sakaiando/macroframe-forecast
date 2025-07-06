# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "macroframe-forecast"
copyright = "2024-2025, IMF"
author = "Ando Sakai, Shuvam Das, Sultan Orazbayev"
release = "0.1.5"


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

html_theme = "furo"
html_static_path = []
