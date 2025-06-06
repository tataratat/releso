# Reinforcement Learning based Shape Optimization (ReLeSO)

[![Read the docs](https://readthedocs.org/projects/releso/badge/?version=latest)](https://releso.readthedocs.io/en/latest/?badge=latest)
[![PyPI - Version](https://img.shields.io/pypi/v/releso)](https://pypi.org/project/releso/)
[![Python Versions](https://img.shields.io/pypi/pyversions/releso)](https://pypi.org/project/releso/)
[![PyPI - License](https://img.shields.io/pypi/l/releso)](https://github.com/tataratat/releso/blob/main/LICENSE)

Releso is a Library/Framework for
Reinforcement Learning based Shape Optimization. Please look into the
Documentation for information on how it works. The instruction on how the
documentation can be built is given below as well as the instruction on how the
package can be installed.
Alternatively, it can be installed from `pip` via `pip install releso`.


Documentation generation
========================

Install and usage instructions are provided in the documentation of the
project. The documentation can be built with the use of sphinx which is a `python`
tool to generate documentation.
> The sphinx packages can either be installed in the project python environment
or in a separate environment. If it does not matter in which python environment
sphinx is installed ignore the first two lines.

The following command line calls create a conda environment with all necessary
dependencies for building the documentation.
``` console
(base) $ conda create -n sphinx python=3.11
(base) $ conda activate sphinx
(sphinx) $ pip install ".[docs]"
```

The documentation is built by executing the following command inside the folder
`docs/`. After executing the command the documentation should be available
inside the folder [`docs/build/html/`](docs/build/html)

``` console
(sphinx) $ make html
```

Installation
============

This section covers the installation process of the framework and its
prerequisites. The first thing to note is that with version 0.1.0 the strict
dependency on `splinepy` is not present anymore. But if the
geometry is to be parameterized by a Spline and the method of Free Form
Deformation is to be used to deform a mesh, `splinepy` is still
necessary.

Prerequisites
-------------
To use `ReLeSO` the following packages have to be installed:
 - pydantic<2
 - stable-baselines3
 - tensorboard
 - hjson

 > The `pydantic` package currently needs to be on version 1.\*, we welcome
 anyone wanting to update `releso` to the new `pydantic` version.

The packages can be installed via pip or conda with the following commands:

**pip** (activation of the venv should be done beforehand)

``` console
(.venv) $ pip install pydantic stable-baselines3 tensorboard hjson
```

**conda**

``` console
(base) $ conda create -n releso python=3.11 "pydantic<2" tensorboard
(base) $ conda activate releso
(releso) $ pip install stable-baselines3 hjson
```

If the spline-based shape optimization functionality is needed, the package
``splinepy`` is needed. Please visit
[`splinepy` on github](https://github.com/tataratat/splinepy) for installation
instructions.

**Development**

To develop the framework further the sphinx package should also be installed
with the currently used sphinx html theme ``sphinx_rtd_theme``.
This can be done via:

``` console
(releso) $ pip install sphinx sphinx_rtd_theme
```

Framework
---------

After installing all prerequisites the framework itself can be installed by
running the command below in the main repository folder.

**Non-development**

```console
(releso) $ pip install .
```

**Development**

``` console
(releso) $ pip install -e ".[dev]"
(releso) $ pip install pre-commit
(releso) $ pre-commit install
```
