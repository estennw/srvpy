import setuptools

VERSION = "0.0.1"
AUTHOR = "Esten Nicolai Wøien"
EMAIL = "esten.n.woien@ntnu.no"


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="srvpy",
    version="0.0.1",
    author=AUTHOR,
    author_email=EMAIL,
    description="A collection of tools for shape analysis in the Square Root Velocity Framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.com/estennw/srvpy",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires='>=3.7',
)
