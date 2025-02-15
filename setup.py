import os
from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize


def get_cython_extensions():
    """
    Walk through the cnake_charmer/cy directory (and its subdirectories)
    and collect all .pyx files to compile as Cython extensions.
    """
    extensions = []
    for root, _, files in os.walk(os.path.join("cnake_charmer", "cy")):
        for file in files:
            if file.endswith(".pyx"):
                file_path = os.path.join(root, file)
                # Convert file path to module name:
                # e.g., "cnake_charmer/cy/math/add.pyx" => "cnake_charmer.cy.math.add"
                module_name = file_path.replace(os.path.sep, ".").replace(".pyx", "")
                extensions.append(Extension(module_name, [file_path]))
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
        get_cython_extensions(),
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
