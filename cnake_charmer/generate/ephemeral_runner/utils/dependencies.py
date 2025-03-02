"""
Dependency detection and management utilities.

This module provides functions for detecting and managing dependencies in code.
"""

import re
import logging
from typing import List, Set, Dict, Any

# Configure logger
logger = logging.getLogger("ephemeral_runner.utils.dependencies")

# Known Python standard library modules that don't need to be installed
PYTHON_STDLIB = {
    # Basic Python modules
    "sys",
    "os",
    "typing",
    "re",
    "subprocess",
    "traceback",
    "math",
    "time",
    # Collections and data structures
    "collections",
    "array",
    "dataclasses",
    "enum",
    "heapq",
    "queue",
    "bisect",
    # Threading and concurrency
    "threading",
    "multiprocessing",
    "concurrent",
    "asyncio",
    "_thread",
    # IO and file handling
    "io",
    "pathlib",
    "tempfile",
    "shutil",
    "fileinput",
    # Data format handling
    "json",
    "csv",
    "pickle",
    "shelve",
    "sqlite3",
    "xml",
    "html",
    # Network and internet
    "socket",
    "ssl",
    "http",
    "urllib",
    "ftplib",
    "poplib",
    "imaplib",
    "smtplib",
    "email",
    # Date and time
    "datetime",
    "calendar",
    "zoneinfo",
    # Text processing
    "string",
    "textwrap",
    "difflib",
    "unicodedata",
    # Others
    "random",
    "itertools",
    "functools",
    "contextlib",
    "abc",
    "argparse",
    "copy",
    "hashlib",
    "logging",
    "platform",
    "uuid",
    "weakref",
}

# System C libraries that don't need to be installed
SYSTEM_LIBS = {"libc", "cpython", "libcpp", "posix"}

# Common library aliases and their actual package names
COMMON_ALIASES = {
    "np": "numpy",
    "pd": "pandas",
    "plt": "matplotlib",
    "tf": "tensorflow",
    "torch": "torch",
    "sk": "scikit-learn",
    "sp": "scipy",
}


def parse_imports(code_str: str, is_cython: bool = False) -> List[str]:
    """
    Parse import statements from code to determine dependencies.

    Args:
        code_str: Code to parse
        is_cython: Whether the code is Cython

    Returns:
        List of dependency names
    """
    libs = set()

    try:
        # Pattern for regular imports and cimports
        import_pattern = re.compile(
            r"^(?:cimport|import|from)\s+([a-zA-Z0-9_\.]+)", re.MULTILINE
        )
        matches = import_pattern.findall(code_str)
        logger.debug(f"Found {len(matches)} import statements in code")

        for m in matches:
            top_level = m.split(".")[0]
            if top_level not in PYTHON_STDLIB and top_level not in SYSTEM_LIBS:
                libs.add(top_level)
                logger.debug(f"Added '{top_level}' to dependencies")

        # Check for common library aliases that might not be explicitly imported
        if is_cython:
            for alias, lib_name in COMMON_ALIASES.items():
                # Look for usage patterns like "np." or "pd." or "np.array"
                if f"{alias}." in code_str and lib_name not in libs:
                    logger.debug(
                        f"Detected potential {lib_name} usage via '{alias}' alias"
                    )
                    libs.add(lib_name)

        # Convert set to a sorted list
        return sorted(libs)
    except Exception as e:
        logger.error(f"Error parsing dependencies: {str(e)}")
        # Return a minimal set to avoid complete failure
        return ["cython"] if is_cython else []


def detect_cython(code_str: str) -> bool:
    """
    Detect whether the given code is Cython.

    Args:
        code_str: Code to check

    Returns:
        True if the code is Cython, False otherwise
    """
    low = code_str.lower()
    cython_indicators = ["cdef", ".pyx", "cimport", "cython"]

    # Check each indicator
    found_indicators = [
        ind
        for ind in cython_indicators
        if ind in (code_str if ind != "cython" else low)
    ]

    if found_indicators:
        logger.debug(f"Identified as Cython due to: {', '.join(found_indicators)}")
        return True
    return False
