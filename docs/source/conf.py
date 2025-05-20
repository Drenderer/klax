# Copyright 2025 The Klax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
copyright = "2025, The Klax Authors."
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



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]

html_theme_options = {
    "use_repository_button": True,
    "repository_url": "https://github.com/Drenderer/klax",
}

html_title = "Klax Docs"

html_logo = "_static/dummy_logo.png"


# -- Autodoc configuration  -------------------------------------------------
autodoc_typehints_description_target = "all"

# Aliases for some lengthy type annotations
autodoc_type_aliases = {
    "ArrayLike": "jaxtyping.ArrayLike",
    "PRNGKeyArray": "PRNGKeyArray",
    "DataTree": "DataTree",
    "MaskTree": "MaskTree"
}

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": True,
    "exclude-members": "__weakref__, __delattr__, __setattr__",
}


# -- Intershinx configuration -------------------------------------------------
# InterSphinx configuration
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
    "jaxtyping": ("https://docs.kidger.site/jaxtyping", None),
    "equinox": ("https://docs.kidger.site/equinox", None),
    "optax": ("https://optax.readthedocs.io/en/latest/", None),
    "paramax": ("https://danielward27.github.io/paramax/", None),
}