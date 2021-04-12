import setuptools

#with open("README.md", "r") as fh:
#    long_description = fh.read()

long_description = ""

setuptools.setup(
    name="srvpy", # Replace with your own username
    version="0.0.1",
    author="Esten Nicolai Wøien",
    author_email="esten.n.woien@ntnu.no",
    description="A package for shape analysis using the Square Root Velocity Framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/estennw/srvpy",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
