# Spline based Shape Optimization via Reinforcement Learning

This repository holds a Library/Framework written by Clemens Fricke for Spline based Shape Optimization via Reinforcement Learning. Please look into the Documentation for information on how it works. The instruction on how the documentation can be build is given below as well as the instruction on how the package can be installed. It is currently not available from pip, this might come in the future.


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

This page covers the installation process of the framework and its prerequisites.

Prerequisites
-------------
To use SbSOvRL the following packages have to be installed:
 - pydantic
 - stable-baselines3
 - tensorboard
 - torchvision
 - hjson
 - gustav

The first two/three packages can be installed via pip and/or conda with the following command:

**pip** (activation of the venv should be done beforehand)

``` console
(.venv) $ pip install pydantic stable-baselines3 tensorboard hjson
```

**conda**

``` console
(base) $ conda create -n SbSOvRL python=3.9 pydantic tensorboard
(base) $ conda activate SbSOvRL
(SbSOvRL) $ conda install -c pytorch torchvision
(SbSOvRL) $ pip install stable-baselines3 hjson
```

The next step is to install the ``gustav`` package which is a python interface for the c++ library SplineLib.
To install ``gustav`` the following repository must be downloaded into an external folder and installed into the venv or conda environment as before. The installation process for the gustav package is documented in the README file of the repository.


**Development**

To develop the framework further the sphinx package should also be installed with the currently used sphinx html theme ``sphinx_rtd_theme``.
The this can be done via:

``` console
(SbSOvRL) $ pip install sphinx sphinx_rtd_theme
```

Framework
---------

After installing all prerequisites the framework itself can be installed by running the command below in the main repository folder

**Non-development**

```console
(SbSOvRL) $ pip install .
```

**Development**

``` console
(SbSOvRL) $ pip install -e .
```
