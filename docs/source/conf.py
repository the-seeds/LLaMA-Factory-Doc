# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "LlamaFactory"
copyright = "2024, LlamaFactory team."
author = "Ziyang Miao, Yaowei Zheng"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["myst_parser"]
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
templates_path = ["_templates"]
exclude_patterns = []
locale_dirs = ['../locales/']
gettext_uuid = True
gettext_compact = False
# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_title = "LlamaFactory"
html_theme = "furo"
html_static_path = ["_static"]
html_css_files = [
    'custom.css',
]