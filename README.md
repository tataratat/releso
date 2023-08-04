# Reinforcement Learning based Shape Optimization

This repository holds a Library/Framework written by Clemens Fricke for Reinforcement Learning based Shape Optimization. Please look into the Documentation for information on how it works. The instruction on how the documentation can be build is given below as well as the instruction on how the package can be installed. It is currently not available from pip, this might come in the future.


Documentation generation
========================

Install and usage instructions are provided in the documentation of the project. The documentation can build with the use of sphinx which is a python tool to generate documentation.
> The sphinx packages can either be installed in the project python environment or in a separate environment. If it does not matter in which python environment sphinx is installed ignore the first two lines.

The following command line calls create a conda environment with all necessary dependency for building the documentation.
``` console
 (base) $ conda create -n sphinx python=3.9
 (base) $ conda activate sphinx
 (sphinx) $ pip install sphinx sphinx-rtd-theme
```

The documentation is build by executing the following command inside the folder "docs/". After executing the command the documentation should be available inside the folder ["docs/build/html/"](docs/build/html)
``` console
(sphinx) $ make html
```

Installation
============

This page covers the installation process of the framework and its prerequisites. The first thing to note is that with version 1.0 the strict dependency on gustaf is not present anymore. But if the geometry is to be parameterized by a Spline and the method of Free Form Deformation is to be used to deform a mesh, gustaf is necessary.

Prerequisites
-------------
To use SbSOvRL the following packages have to be installed:
 - pydantic<2
 - stable-baselines3
 - tensorboard
 - hjson

 > The *pydantic* package currently needs to be on version 1.\*, we welcome anyone wanting to update *releso* to the new *pydantic* version.

The packages can be installed via pip or conda with the following commands:

**pip** (activation of the venv should be done beforehand)

``` console
(.venv) $ pip install pydantic stable-baselines3 tensorboard hjson
```

**conda**

``` console
(base) $ conda create -n releso python=3.9 pydantic tensorboard
(base) $ conda activate releso
(releso) $ pip install stable-baselines3 hjson
```

If the spline based shape optimization functionality is needed, the package ``gustaf`` is needed. Please visit [gustaf on github](https://github.com/tataratat/gustaf) for installation instructions.


**Development**

To develop the framework further the sphinx package should also be installed with the currently used sphinx html theme ``sphinx_rtd_theme``.
The this can be done via:

``` console
(releso) $ pip install sphinx sphinx_rtd_theme
```

Framework
---------

After installing all prerequisites the framework itself can be installed by running the command below in the main repository folder

**Non-development**

```console
(releso) $ pip install .
```

**Development**

``` console
(releso) $ pip install -e .
```
