# Spline based Shape Optimization via Reinforcement Learning

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

***
<div style="color: gray">
# Old README

1. Install all necessary packages
    - Install packages in requirements.txt
    - install Gustav from the git repository
    - make sure the solver is runnable
2. Prepare Solver files
    a. A deviation from the norm is the xns.in.mesh. Here the nrng and nprm files should not point to the deformed mesh files but the corresponding files of the original mesh. This alleviates the problem of repartitioning the mesh in every step and since the connectivity is unchanged the partitioning should also be unchanged.
3. Fix Gustav. Currently Gustav has still a little bug the prevents me from using it correctly. in load.py in line ~88 the check if the path is absolute must be commented. (actually should be fix be the time you read this)

If developing this package one can use ```pip install -e .``` in the 05-Code/SORL folder to install a development version where you can use  ```import SbSOvRL``` where ever you are (as long as the python environment is correctly set). You only have to do this once since it will link to the development folder and not to ``site-packages``. 
</div>