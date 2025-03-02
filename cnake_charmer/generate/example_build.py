#!/usr/bin/env python3
"""
Example demonstrating how to use:
1. The ephemeral_runner for both Python and Cython code
2. The Cython analyzer for performance optimization analysis
3. The reward system for evaluating code quality

This script:
1. Defines sample Python and Cython code snippets
2. Uses the appropriate builders to build and test each snippet
3. Analyzes the Cython code for optimization opportunities
4. Evaluates code quality using the reward system
"""

import logging
import os

# Configure logging to show info level information
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Import the builders
from ephemeral_runner.builders import PythonBuilder, CythonBuilder

# Import the Cython analyzer components
from cython_analyzer import (
    CythonAnalyzer,
    get_optimization_report,
    get_optimization_hints,
    get_analysis_explanation,
)

# Import the Cython reward system creator
from cython_analyzer.reward import create_cython_reward_system


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

    # Define sample Cython code with type annotations - FIXED with malloc/free for nogil compatibility
    optimized_cython_code = """# cython: boundscheck=False
# cython: wraparound=False

import numpy as np
cimport numpy as np
from cython.parallel import prange
from libc.stdlib cimport malloc, free

# Using C malloc/free for nogil-compatible memory management
cdef long long fibonacci_cy_impl(int n) nogil:
    if n <= 1:
        return n
    if n == 2:
        return 1
        
    # Use C allocation instead of numpy for nogil compatibility
    cdef long long* fib = <long long*>malloc((n + 1) * sizeof(long long))
    if fib == NULL:
        # Handle allocation failure
        with gil:
            raise MemoryError("Failed to allocate memory")
    
    # Initialize values
    fib[1] = 1
    fib[2] = 1
    
    # Calculate Fibonacci sequence
    cdef int i
    for i in prange(3, n + 1):
        fib[i] = fib[i-1] + fib[i-2]
    
    # Store result before freeing memory
    cdef long long result = fib[n]
    free(fib)
    
    return result

# Python-accessible wrapper function
def fibonacci_cy(int n):
    if n <= 0:
        raise ValueError("Input must be a positive integer")
    return fibonacci_cy_impl(n)

def fibonacci_sequence_cy(int n):
    cdef int i
    cdef np.ndarray[np.int64_t, ndim=1] result = np.zeros(n, dtype=np.int64)
    
    for i in range(n):
        result[i] = fibonacci_cy(i+1)
        
    return result
"""

    # Define a less optimized version of the Cython code
    unoptimized_cython_code = """# Simple Cython implementation
import numpy as np
cimport numpy as np

def fibonacci_cy(n):
    if n <= 0:
        raise ValueError("Input must be a positive integer")
    if n == 1 or n == 2:
        return 1
        
    # Use dynamic programming to calculate Fibonacci
    fib = np.zeros(n + 1, dtype=np.int64)
    fib[1] = fib[2] = 1
    
    for i in range(3, n + 1):
        fib[i] = fib[i-1] + fib[i-2]
        
    return fib[n]

def fibonacci_sequence_cy(n):
    result = np.zeros(n, dtype=np.int64)
    
    for i in range(n):
        result[i] = fibonacci_cy(i+1)
        
    return result
"""

    print("\n" + "=" * 80)
    print("SECTION 1: BUILDING AND TESTING PYTHON CODE")
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
    print("SECTION 2: BUILDING AND TESTING CYTHON CODE")
    print("=" * 80)

    # Create a Cython builder with a specific request ID
    request_id = "example_cython_run"
    cython_builder = CythonBuilder(request_id=request_id)

    # Build and run the optimized Cython code
    print("\nBuilding and running optimized Cython code...")
    result = cython_builder.build_and_run(optimized_cython_code)

    if not result.success:
        print(f"Error building and running Cython code:\n{result.error_message}")
    else:
        print("Optimized Cython code built and ran successfully!")

    print("\n" + "=" * 80)
    print("SECTION 3: ANALYZING CYTHON CODE OPTIMIZATION")
    print("=" * 80)

    # Create a new builder specifically for analysis
    analysis_request_id = "analysis_run"
    analysis_builder = CythonBuilder(request_id=analysis_request_id)

    # Create a CythonAnalyzer instance that will use our builder
    analyzer = CythonAnalyzer(ephemeral_runner=analysis_builder)

    print("\nAnalyzing optimized Cython code...")
    # Using the structured analysis method
    analysis_result = analyzer.analyze_code_structured(optimized_cython_code)

    # Generate a detailed report
    print("\nDetailed Optimization Report for Optimized Code:")
    print(get_analysis_explanation(analysis_result))

    # Get and display specific optimization hints
    hints = get_optimization_hints(analysis_result)
    if hints:
        print("\nLine-specific optimization hints:")
        for line_num, hint in sorted(hints.items()):
            print(f"Line {line_num}: {hint}")
    else:
        print("\nNo specific optimization hints found.")

    # Also analyze the unoptimized version for comparison
    print("\nAnalyzing unoptimized Cython code...")
    unoptimized_analysis = analyzer.analyze_code_structured(unoptimized_cython_code)

    print("\nOptimization Score Comparison:")
    print(f"- Optimized code: {analysis_result.optimization_score:.2f}")
    print(f"- Unoptimized code: {unoptimized_analysis.optimization_score:.2f}")
    print(
        f"- Improvement: {analysis_result.optimization_score - unoptimized_analysis.optimization_score:.2f} points"
    )

    print("\n" + "=" * 80)
    print("SECTION 4: EVALUATING CODE WITH REWARD SYSTEM")
    print("=" * 80)

    # Create a reward system specifically for Cython code evaluation
    print("\nCreating reward system for Cython evaluation...")
    reward_system = create_cython_reward_system()

    # Set up inputs and outputs for reward calculation
    inputs = {"prompt": "Create an efficient Cython function for Fibonacci numbers"}

    # Evaluate optimized code
    print("\nEvaluating optimized Cython code with reward system...")
    outputs_optimized = {"generated_code": optimized_cython_code}
    optimized_score = reward_system.calculate_reward(inputs, outputs_optimized)

    print("\nReward System Evaluation for Optimized Code:")
    print(reward_system.get_score_explanation())

    # Evaluate unoptimized code
    print("\nEvaluating unoptimized Cython code with reward system...")
    outputs_unoptimized = {"generated_code": unoptimized_cython_code}
    unoptimized_score = reward_system.calculate_reward(inputs, outputs_unoptimized)

    print("\nReward System Evaluation for Unoptimized Code:")
    print(reward_system.get_score_explanation())

    print("\nReward System Score Comparison:")
    print(f"- Optimized code score: {optimized_score:.2f}")
    print(f"- Unoptimized code score: {unoptimized_score:.2f}")
    print(f"- Improvement: {optimized_score - unoptimized_score:.2f} points")


if __name__ == "__main__":
    main()
