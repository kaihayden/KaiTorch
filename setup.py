import setuptools

with open("README.md", "r") as readme:
    long_description = readme.read()

setuptools.setup(
    name="kaitorch",
    version="0.1.0",
    author="Kai Hayden",
    author_email="kaihayden97@gmail.com",
    description="A Keras-like neural network library with an autograd engine operating on a dynamically built DAG of scalar values",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kaihayden/kaitorch",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)