# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../../src/'))


# -- Project information -----------------------------------------------------

project = 'SbSOvRL'
copyright = '2022, Clemens Fricke'
author = 'Clemens Fricke'

# The full version, including alpha/beta/rc tags
release = '1.1'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon'
]
autosummary_generate = True  # Turn on sphinx.ext.autosummary
# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']
autodoc_mock_imports = ["pydantic", "tensorboard", "hjson", "gym",
                        "stable_baselines3", "pandas", "gustav", "numpy",
                        "vedo", "matplotlib", "imageio"]
# autodoc_typehints = 'description'  # show type hints in doc body instead of signature
# autoclass_content = 'instance'  # get docstring from class level and init simultaneously

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'
html_logo = '_static/logo.png'
html_favicon = '_static/thumb.png'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
