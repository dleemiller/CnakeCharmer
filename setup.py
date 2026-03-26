import os
from typing import Literal

from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup

SIMD_COMPILE_ARGS = ["-mavx2", "-mfma", "-O3"]
SIMD_CATEGORIES = {"nn_ops"}


def get_cython_extensions(syntax: Literal["cy", "pp"]):
    """
    Walk through the cnake_charmer/cy directory (and its subdirectories)
    and collect all .pyx files to compile as Cython extensions.
    """
    extensions = []
    file_extension = ".pyx" if syntax == "cy" else ".py"
    for root, _, files in os.walk(os.path.join("cnake_charmer", syntax)):
        for file in files:
            if file.endswith(file_extension):
                file_path = os.path.join(root, file)
                module_name = file_path.replace(os.path.sep, ".").replace(file_extension, "")
                # Add SIMD flags for nn_ops category
                category = os.path.basename(root)
                extra_args = SIMD_COMPILE_ARGS if category in SIMD_CATEGORIES else []
                extensions.append(
                    Extension(
                        module_name,
                        [file_path],
                        extra_compile_args=extra_args,
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
    packages=find_packages(),
    # ext_modules=extensions,
    ext_modules=cythonize(
        get_cython_extensions(syntax="cy") + get_cython_extensions(syntax="pp"),
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
