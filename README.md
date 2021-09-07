# Spline based Shape Optimization via Reinforcement Learning


# Stuff to do before you can use this tool

1. Install all necessary packages
    - Install packages in env.yml
    - install Gustav from the git repository
    - make sure the solver is runnable
2. Prepare Solver files
    a. A deviation from the norm is the xns.in.mesh. Here the nrng and nprm files should not point to the deformed mesh files but the coresponding files of the original mesh. This eleviates the problem of repartitioning the mesh in every step and since the connectivity is unchanged the partitioning should also be unchanged.
3. Fix Gustav. Currently Gustav has still a little bug the prevents me from using it correctly. in load.py in line ~88 the check if the path is absolute must be commented. (actually should be fix be the time you read this)

If developing this package one can use ```pip install -e .``` in the 05-Code/SORL folder to install a development version where you can use  ```import SbSOvRL``` where ever you are (as long as the python environment is correctly set). You only have to do this once since it will link to the development folder and not to ``site-packages``. 