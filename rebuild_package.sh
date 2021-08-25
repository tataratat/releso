# this only works if package with version 0.0.1 is already installed and version is still at 0.0.1


# build the package
python -m build

# install package 
pip install dist/SbSOvRL-0.0.1-py3-none-any.whl --force-reinstall