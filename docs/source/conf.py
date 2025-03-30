# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import sys
from pathlib import Path


# sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path("..").resolve()))
print(Path("..").resolve())


project = "Klax"
copyright = "2025, Jasper Schommartz, Fabian Roth"
author = "Jasper Schommartz, Fabian Roth"
release = "0.1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
]

templates_path = ["_templates"]
exclude_patterns = []

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
    "jaxtyping": ("https://docs.kidger.site/jaxtyping", None),
    "equinox": ("https://docs.kidger.site/equinox", None),
    "optax": ("https://optax.readthedocs.io/en/latest/", None),
    "paramax": ("https://danielward27.github.io/paramax/", None),
}


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]

html_theme_options = {
    "use_repository_button": True,
    "repository_url": "https://github.com/Drenderer/klax",
}

html_title = "Klax Docs"

autodoc_typehints_description_target = "all"
autodoc_type_aliases = {
    "ArrayLike": "jaxtyping.ArrayLike",
    "PRNGKeyArray": "PRNGKeyArray",
    "DataTree": "DataTree",
    "MaskTree": "MaskTree"
}
