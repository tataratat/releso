"""Definition of the package and installation of it."""
import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="SbSOvRL",
    version="1.1.0",
    author="Clemens Fricke",
    author_email="clemens.fricke@ilsb.tuwien.ac.at",
    description="""
    Spline based Shape Optimization via Reinforcement Learning is a framework
    and library which implements a new way to optimize geometries with
    Reinforcement Learning. The parameterization of the geometry is done via
    Splines. Please check in the documentation for more information on how this
     library functions.

    The authors also wrote a seminar and master thesis during the development
    of this library in which the underling functionality is given. If this is
    needed please contact either clemens.fricke@rwth-aachen.de,
    clemens.fricke@ilsb.tuwien.ac.at or wolff@cats.rwth-aachen.de.
    """,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    project_urls={},
    classifiers=[
        "Programming Language :: Python :: 3",
        # "License :: OSI Approved :: MIT License",
        # "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.8",
)
