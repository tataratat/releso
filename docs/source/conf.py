"""Configuration file for the Sphinx documentation builder.

This file only contains a selection of the most common options. For a full
list see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html

-- Path setup --------------------------------------------------------------

If extensions (or modules to document with autodoc) are in another directory,
add these directories to sys.path here. If the directory is relative to the
documentation root, use os.path.abspath to make it absolute, like shown here.


"""
from pydantic import BaseModel
from sphinx.ext.napoleon import _skip_member

from releso.__version__ import version

# sys.path.insert(0, str(releso_dir / "util"))
# -- Project information -----------------------------------------------------

project = "ReLeSO"
copyright = "2023, Clemens Fricke"  # noqa: A001
author = "Clemens Fricke"

# The full version, including alpha/beta/rc tags
release = version


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx-jsonschema",
    # 'sphinxcontrib.autodoc_pydantic'
]
autosummary_generate = True  # Turn on sphinx.ext.autosummary
# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]
autodoc_mock_imports = [
    "gymnasium",
    "gustaf",
    "vedo",
    "matplotlib",
    "imageio",
    "torchvision",
    "splinepy",
]
# show type hints in doc body instead of signature
# autodoc_typehints = "both"
# get docstring from class level and init simultaneously
# autoclass_content = 'instance'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# autodoc_pydantic_model_show_json = True
# autodoc_pydantic_settings_show_json = False

# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"
html_logo = "_static/logo.png"
html_favicon = "_static/thumb.png"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]


# static variable needs only be initialized once
pydantic_functions = list(BaseModel.__dict__)


def skip(app, what, name, obj, would_skip, options):
    """Check if member should be shown in the documentation.

    A little complication, is that if a `autodoc-skip-member` event is
    registered if the `napoleon` extension is enabled, it will be called
    instead of the default skip member function. (It is not additive)
    So I am calling this function from this new skip function so that I do
    not have to reimplement the same functionality.

    Additional members that are skipped are pydantic functions that are
    inherited and functions that start with `check_` or `validate_`. These
    functions are used as validators for the data in this package and should
    normally not be called from user code.

    Args:
        app (sphinx.application.Sphinx): Sphinx application.
        what (str): Type of the object.
        name (str): Name of the Object.
        obj (Any): The object itself.
        would_skip (bool): Decision of the object should be skipped from
            previous checks.
        options (Any): Options.

    Returns:
        bool: Whether or not to skip the object.
    """
    if would_skip := _skip_member(app, what, name, obj, would_skip, options):
        return would_skip
    if name in pydantic_functions:
        return True
    if name.startswith(("check_", "validate_")):
        return True
    # return would_skip


def setup(app):
    """Method that is called from sphinx to load user code for the docs build.

    Args:
        app (sphinx.application.Sphinx): Sphinx application.
    """
    app.connect("autodoc-skip-member", skip)
