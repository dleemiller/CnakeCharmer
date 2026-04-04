import os
from typing import Literal

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup

NUMPY_INCLUDE = [np.get_include()]

OPENMP_COMPILE_ARGS = ["-fopenmp"]
OPENMP_LINK_ARGS = ["-fopenmp"]


def _uses_openmp(file_path):
    """Check if a .pyx file imports cython.parallel (prange, parallel)."""
    try:
        with open(file_path) as f:
            for line in f:
                if "cython.parallel" in line or "from cython.parallel" in line:
                    return True
    except OSError:
        pass
    return False


def _uses_cpp(file_path):
    """Check if a .pyx file needs C++ compilation (libcpp, cppclass, directive)."""
    try:
        with open(file_path) as f:
            for line in f:
                if "from libcpp" in line or "cdef cppclass" in line:
                    return True
                if "distutils: language" in line and "c++" in line:
                    return True
    except OSError:
        pass
    return False


def _uses_pythran(file_path):
    """Check if a .pyx file uses Pythran's NumPy backend."""
    try:
        with open(file_path) as f:
            for line in f:
                if "np_pythran" in line and "True" in line:
                    return True
    except OSError:
        pass
    return False


def _get_pythran_include():
    """Get Pythran's include directory, or empty list if unavailable."""
    try:
        import pythran.config

        return [pythran.config.get_include()]
    except (ImportError, AttributeError):
        return []


def get_cython_extensions(syntax: Literal["cy"]):
    """
    Walk through the cnake_charmer/{syntax} directory and collect
    all .pyx or .py files to compile as Cython extensions.
    """
    extensions = []
    file_extension = ".pyx"
    for root, _, files in os.walk(os.path.join("cnake_data", syntax)):
        for file in files:
            if file.endswith(file_extension):
                file_path = os.path.join(root, file)
                module_name = file_path.replace(os.path.sep, ".").replace(file_extension, "")
                extra_compile = []
                extra_link = []
                include_dirs = list(NUMPY_INCLUDE)
                lang = None
                if _uses_openmp(file_path):
                    extra_compile = extra_compile + OPENMP_COMPILE_ARGS
                    extra_link = OPENMP_LINK_ARGS
                if _uses_cpp(file_path):
                    lang = "c++"
                if _uses_pythran(file_path):
                    include_dirs = include_dirs + _get_pythran_include()
                extensions.append(
                    Extension(
                        module_name,
                        [file_path],
                        include_dirs=include_dirs,
                        extra_compile_args=extra_compile,
                        extra_link_args=extra_link,
                        language=lang,
                    )
                )

    return extensions


setup(
    name="CnakeCharmer",
    version="0.1.0",
    description=(
        "A living dataset of 10,000 equivalent Python/Cython implementations "
        "for AI translation training and benchmarking."
    ),
    author="Lee Miller",
    author_email="dleemiller@protonmail.com",
    url="https://github.com/dleemiller/CnakeCharmer",
    packages=find_packages(
        include=["cnake_charmer", "cnake_charmer.*", "cnake_data", "cnake_data.*"]
    ),
    # ext_modules=extensions,
    ext_modules=cythonize(
        get_cython_extensions(syntax="cy"),
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
        },
        annotate=True,
    ),
    install_requires=[
        "numpy>=2.0.0",
        "cython>=3.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-benchmark>=3.2.3",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Cython",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)
