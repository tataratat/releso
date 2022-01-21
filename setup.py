import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="SbSOvRL",
    version="0.0.1",
    author="Clemens Fricke",
    author_email="clemens.fricke@rwth-aachen.de",
    description="Spline based Shape Optimization via Reinforcement Learning uses Splines to deform the mesh that represents the shape that is to be optimized.",
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