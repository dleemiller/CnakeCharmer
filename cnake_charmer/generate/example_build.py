#!/usr/bin/env python3
"""
Example demonstrating how to use:
1. The ephemeral_runner for both Python and Cython code
2. The Cython analyzer for performance optimization analysis

This script:
1. Defines sample Python and Cython code snippets
2. Uses the appropriate builders to build and test each snippet
3. Analyzes the Cython code for optimization opportunities
4. Shows the output of the build, test, and analysis processes
"""

import logging
import os

# Configure logging to show debug information
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Import the builders
from ephemeral_runner.builders import PythonBuilder, CythonBuilder

# Import the Cython analyzer components
from cython_analyzer import (
    CythonAnalyzer,
    get_optimization_report,
    get_optimization_hints,
)


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

    # Create a Cython builder with a specific request ID
    request_id = "example_cython_run"
    cython_builder = CythonBuilder(request_id=request_id)

    # Build and run the Cython code (without annotation for this run)
    print("\nBuilding and running Cython code...")
    result = cython_builder.build_and_run(cython_code)

    if not result.success:
        print(f"Error building and running Cython code:\n{result.error_message}")
    else:
        print("Cython code built and ran successfully!")

    print("\n" + "=" * 80)
    print("EXAMPLE 3: ANALYZING CYTHON CODE FOR OPTIMIZATION")
    print("=" * 80)

    # Create a new builder specifically for analysis
    analysis_request_id = "analysis_run"
    analysis_builder = CythonBuilder(request_id=analysis_request_id)

    # Create a CythonAnalyzer instance that will use our builder
    analyzer = CythonAnalyzer(ephemeral_runner=analysis_builder)

    print("\nAnalyzing Cython code optimization...")
    # Analyze the Cython code
    metrics = analyzer.analyze_code(cython_code)

    # Generate an optimization report
    report = get_optimization_report(metrics)
    print("\nCython Optimization Report:")
    print(report)

    # Get specific optimization hints
    hints = get_optimization_hints(metrics)
    if hints:
        print("\nLine-specific optimization hints:")
        for line_num, hint in sorted(hints.items()):
            print(f"Line {line_num}: {hint}")
    else:
        print("\nNo specific optimization hints found.")

    # Print the overall optimization score
    print(f"\nOverall optimization score: {metrics.get('optimization_score', 0):.2f}")
    print("(Higher score indicates better Cython optimization)")


if __name__ == "__main__":
    main()
