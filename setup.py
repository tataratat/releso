"""Definition of the package and installation of it."""
import setuptools

with open("releso/_version.py") as f:
    version = eval(f.read().strip().split("=")[-1])

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="releso",
    version=version,
    author="Clemens Fricke",
    author_email="clemens.david.fricke@tuwien.ac.at",
    description="""
    Reinforcement Learning based Shape Optimization (releso) is a framework
    and library which implements a new way to optimize geometries with
    Reinforcement Learning. The parameterization of the geometry is usually
    done with spline, but it is not necessary.

    Please check in the documentation for more information on how this
    library functions.

    The authors also wrote a seminar and master thesis during the development
    of this library in which the underling functionality is given. If this is
    needed please contact either clemens.fricke@rwth-aachen.de,
    clemens.david.fricke@tuwien.ac.at or wolff@cats.rwth-aachen.de.
    """,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    project_urls={},
    classifiers=[
        "Operating System :: OS Independent",
        "License :: OSI Approved :: MIT License",
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Natural Language :: English',
        'Topic :: Scientific/Engineering',
    ],
    license="MIT",
    packages=setuptools.find_packages(),
    python_requires=">=3.8",
    entry_points={'console_scripts': ['releso = releso.__main__:entry']},
)
