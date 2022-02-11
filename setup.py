import os
from setuptools import setup

name = "superposed-pulses"

with open("README.md") as f:
    long_description = f.read()

here = os.path.abspath(os.path.dirname(__file__))

setup(
    name=name,
    description="Python scripts used by the fusion energy group at UiT The Arctic University of Norway.",
    author="Juan Manuel Losada",
    author_email="juan.m.losada@uit.no",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/uit-cosmo/filtered-point-process",
    download_url="https://github.com/uit-cosmo/3ppy/archive/refs/tags/1.1.tar.gz",
    license="MiT",
    version="1.2",
    packages=["model"],
    python_requires=">=3.0",
    install_requires=[
        "numpy>=1.15.0",
        "scipy>=1.4.0",
        "tqdm>=4.50.2",
        "matplotlib>=3.2.0",
    ],
    classifiers=[
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
    zip_safe=False,
)
