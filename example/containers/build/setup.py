# containers/build/setup.py

from setuptools import setup
from Cython.Build import cythonize
import os

filename = os.environ.get("CYTHON_FILE", "fizzbuzz.pyx")

setup(
    name="cython_module",
    ext_modules=cythonize(filename),
)
