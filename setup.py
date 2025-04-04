from setuptools import setup, find_packages
import numpy

setup(
    name="HyHoDy",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        numpy
    ],
)