Installation
============

This page covers the installation process of the framework and its prerequisites. 

Prerequisites
-------------
To use SbSOvRL the following packages have to be installed:
 - pydantic
 - stable-baselines3
 - tensorboard
 - hjson
 - gustav

The first two/three packages can be installed via pip and/or conda with the following command:

**pip** (activation of the venv should be done beforehand)

.. code-block:: console

   (.venv) $ pip install pydantic stable-baselines3 tensorboard hjson

**conda**

.. code-block:: console

   (base) $ conda create -n SbSOvRL python=3.9 pydantic tensorboard
   (base) $ conda activate SbSOvRL
   (SbSOvRL) $ pip install stable-baselines3 hjson

The next step is to install the ``gustav`` package which is a python interface for the c++ library SplineLib.
To install ``gustav`` the following repository must be downloaded into an external folder and installed into the venv or conda environment as before. The installation process for the gustav package is documented in the README file of the repository.


**Development**

To develop the framework further the sphinx package should also be installed and the 

Framework
---------

After installing all prerequisites the framework itself can be installed by running the command below in the repository folder 05-Code/SORL/ 

**Non-development**

.. code-block:: console
  
   (SbSOvRL) $ pip install .


**Development**

.. code-block:: console
   
   (SbSOvRL) $ pip install -e .

