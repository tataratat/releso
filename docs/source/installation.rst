Installation
============

The package is available via PyPI and can be installed via pip:

.. code-block:: console

   (env) $ pip install releso

This command will install the package with the minimal dependencies.

The following packages are optional and bring further capabilities to the framework:
 - splinepy -> Spline based geometries
 - gustaf -> If Free Form Deformations is used
 - torchvision -> If Image based observations are used

These can be automatically install via the following command:

.. code-block:: console

   (env) $ pip install "releso[all]"


.. note::
    It is recommended to use a environment manager like conda to install the packages into an environment.

Installation from source
------------------------

Clone the `repository <https://github.com/clemens-fricke/releso>`_ from github.

There are two modes that a package can be installed from source.

**Non-development**

The package is basically installed like it would from PyPI. This can be done via:

.. code-block:: console

   (releso) $ pip install .


**Development**
The development mode of *pip install* allows to change the source code and have the changed
directly reflected in the installed package. Meaning no recompliation before starting the
next script call is necessary (If you use an IPython kernel you will have to restart the kernel
to see the changes). This is done by adding the ``-e`` flag to the pip install command.
The development mode can be installed via:

.. code-block:: console

   (releso) $ pip install -e ".[dev]"

The def extension will install all optional dependencies as well as the development dependencies.
