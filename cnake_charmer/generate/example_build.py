#!/usr/bin/env python3
"""
Example demonstrating how to use the ephemeral_runner for both Python and Cython code.

This script:
1. Defines sample Python and Cython code snippets
2. Uses the appropriate builders to build and test each snippet
3. Shows the output of the build and test process
"""

import logging
import os

# Configure logging to show debug information
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Import the builders
from ephemeral_runner.builders import PythonBuilder, CythonBuilder


def main():
    # Define sample Python code
    python_code = """
# A simple Python function to calculate Fibonacci numbers
def fibonacci(n):
    if n <= 0:
        raise ValueError("Input must be a positive integer")
    if n == 1 or n == 2:
        return 1
        
    # Use dynamic programming to calculate Fibonacci
    fib = [0] * (n + 1)
    fib[1] = fib[2] = 1
    
    for i in range(3, n + 1):
        fib[i] = fib[i-1] + fib[i-2]
        
    return fib[n]

def fibonacci_sequence(n):
    return [fibonacci(i) for i in range(1, n+1)]
"""

    # Define sample Cython code with type annotations
    cython_code = """# cython: boundscheck=False
# cython: wraparound=False

import numpy as np
cimport numpy as np

def fibonacci_cy(int n):
    if n <= 0:
        raise ValueError("Input must be a positive integer")
    if n == 1 or n == 2:
        return 1
        
    # Use dynamic programming to calculate Fibonacci
    cdef int i
    cdef long long[:] fib = np.zeros(n + 1, dtype=np.int64)
    fib[1] = fib[2] = 1
    
    for i in range(3, n + 1):
        fib[i] = fib[i-1] + fib[i-2]
        
    return fib[n]

def fibonacci_sequence_cy(int n):
    cdef int i
    cdef np.ndarray[np.int64_t, ndim=1] result = np.zeros(n, dtype=np.int64)
    
    for i in range(n):
        result[i] = fibonacci_cy(i+1)
        
    return result
"""

    print("\n" + "=" * 80)
    print("EXAMPLE 1: BUILDING AND TESTING PYTHON CODE")
    print("=" * 80)

    # Create a Python builder
    python_builder = PythonBuilder()

    # Build and run the Python code
    print("\nBuilding and running Python code...")
    error = python_builder.build_and_run(python_code)

    if error:
        print(f"Error building and running Python code:\n{error}")
    else:
        print("Python code built and ran successfully!")

    print("\n" + "=" * 80)
    print("EXAMPLE 2: BUILDING AND TESTING CYTHON CODE")
    print("=" * 80)

    # Create a Cython builder
    cython_builder = CythonBuilder()

    # Build and run the Cython code
    print("\nBuilding and running Cython code...")
    error = cython_builder.build_and_run(cython_code)

    if error:
        print(f"Error building and running Cython code:\n{error}")
    else:
        print("Cython code built and ran successfully!")


if __name__ == "__main__":
    main()
