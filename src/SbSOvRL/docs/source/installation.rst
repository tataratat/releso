Installation
============

This page covers the installation process of the framework and its prerequisites. 

Prerequisites
-------------
To use SbSOvRL the following packages have to be installed:
 - pydantic
 - stable-baselines3
 - tensorboard
 - gustav

The first two/three packages can be installed via pip and/or conda with the following command:

**pip** (activation of the venv should be done beforehand)

.. code-block:: console

   (.venv) $ pip install pydantic stable-baselines3 tensorboard

**conda**

.. code-block:: console

   (base) $ conda create -n SbSOvRL python=3.9 pydantic tensorboard
   (base) $ conda activate SbSOvRL
   (SbSOvRL) $ pip install stable-baselines3

The next step is to install the ``gustav`` package which is a python interface for the c++ library SplineLib.
To install ``gustav`` the following repository must be downloaded into an external folder and installed into the venv or conda environment as before. The installation process for the gustav package is documented in the README file of the repository.

Framework
---------

After installing all prerequisites the framework itself can be installed by running the command 
.. code-block:: console
   
   (SbSOvRL) $ pip install -e .

