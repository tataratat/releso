.. ReLeSO documentation master file, created by
   sphinx-quickstart on Mon Nov 29 15:34:28 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Reinforcement Learning based Shape Optimization (ReLeSO)
========================================================
|Build Status| |Documentation Status|
|PyPI| |Python| |License|

.. |Build Status| image:: https://github.com/tataratat/releso/actions/workflows/build_and_upload_wheels.yml/badge.svg
   :target: https://github.com/tataratat/releso
   :alt: PyPI - Version

.. |Documentation Status| image:: https://readthedocs.org/projects/releso/badge/?version=latest
    :target: https://releso.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. .. |codecov| image:: https://codecov.io/gh/clemensfricke/ReLeSO/branch/master/graph/badge.svg
..     :target: https://codecov.io/gh/clemensfricke/ReLeSO
..     :alt: Code Coverage

.. |PyPI| image:: https://img.shields.io/pypi/v/releso
    :target: https://pypi.org/project/releso/
    :alt: PyPI

.. |Python| image:: https://img.shields.io/pypi/pyversions/releso
    :target: https://pypi.org/project/releso/
    :alt: Python

.. |License| image:: https://img.shields.io/pypi/l/releso
    :target: https://github.com/tataratat/releso/blob/main/LICENSE
    :alt: PyPI - License

**ReLeSO** stands for ``Reinforcement Learning based Shape Optimization`` and is a Python framework combining a spline-based shape optimization approach with reinforcement learning.


This documentation includes the usage information and possible configuration parameters of this framework. Please see the thesis "Python Framework for Reinforcement Learning based Shape Optimization" by Clemens Fricke.  Please contact Clemens Fricke (clemens.david.fricke@tuwien.ac.at) or Daniel Wolff (d.wolff@unibw.de) to access it.
We also released two papers with results obtained with this framework, that also go into the theory of the application of Shape Optimization with Reinforcement Learning. The first paper is a short proceedings about the basic concept of concept of ReLeSO for an introductory example to optimzation of extrusion dies [Wolff2023]_ and the second paper is a more detailed paper about the possible optimization steps towards better learning [Fricke2023]_. In the last paper we compare different agents and the two types of RL-based shape optimization that this framework implements, namely incremental and direct optimization.


This framework is mainly build upon the Python packages ``pydantic`` and ``stable-baselines3``. Especially the RL agents used are from the ``stable-baselines3`` package. So please refer to the documentation of these packages for further information about the agents and a deeper understanding of RL in general. The documentation given `there <https://stable-baselines3.readthedocs.io/en/master/>`_ is very good and easy to understand.


For guidance on the installation process see :doc:`installation`.

.. toctree::
   installation
   usage
   json_schema
   ReLeSO
   SPOR
   feature_extractor
   image_based_observations




Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. [Wolff2023] Wolff, D., Fricke, C., Kemmerling, M., & Elgeti, S. (2023). `Towards shape optimization of flow channels in profile extrusion dies using reinforcement learning <https://onlinelibrary.wiley.com/doi/abs/10.1002/pamm.202200009>`_. PAMM, 22(1), e202200009

.. [Fricke2023] Fricke, C., Wolff, D., Kemmerling, M., & Elgeti, S. (2023). `Investigation of reinforcement learning for shape optimization of 2D profile extrusion die geometries <https://www.aimsciences.org/article/doi/10.3934/acse.2023001>`_. Advances in Computational Science and Engineering, 1(1), 1-35.
