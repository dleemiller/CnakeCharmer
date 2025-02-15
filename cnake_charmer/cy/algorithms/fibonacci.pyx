# distutils: language=c++
# cython: boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True, language_level=3
"""
Fibonacci Module
----------------

This module implements an optimized Fibonacci sequence generator using modern Cython 3
features. It leverages the C++ Standard Template Library's vector for dynamic memory 
management and employs pure Python syntax with type annotations.

The implementation uses an initial fixed capacity with explicit growth strategy when
capacity is exceeded, providing predictable memory management behavior.
"""

import cython
from libcpp.vector cimport vector
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(args=(1e18,))
def fib(n: cython.longlong) -> list[cython.longlong]:
    """Compute all Fibonacci numbers less than n using optimized C++ containers.

    This implementation uses several optimization techniques:
    1. C++ vector for efficient dynamic memory management
    2. Fixed initial capacity with explicit growth strategy
    3. Pure C operations for numeric computations
    4. Memory view optimizations for fast array access
    5. Compiler directive optimizations for maximum performance

    Memory Management:
        - Initial capacity: 100 elements
        - Growth strategy: When capacity is exceeded, new_capacity = capacity + (capacity // 2)
        - Uses C++ vector's reserve() to minimize reallocations

    Args:
        n (cython.longlong): The upper limit for the Fibonacci sequence (exclusive).
            Must be a positive integer.

    Returns:
        list[cython.longlong]: A list containing all Fibonacci numbers less than n,
            in ascending order.

    Example:
        >>> fib(10)
        [1, 1, 2, 3, 5, 8]
        >>> fib(0)
        []
    """
    if n <= 0:
        return []

    # Initialize vector with fixed capacity
    sequence: vector[cython.longlong] = vector[cython.longlong]()
    capacity: cython.size_t = 100
    sequence.reserve(capacity)

    # Generate sequence using pure C operations
    a: cython.longlong = 0
    b: cython.longlong = 1

    while b < n:
        if sequence.size() >= capacity:
            # Grow capacity by 50% when full
            new_capacity: cython.size_t = capacity + (capacity // 2)
            sequence.reserve(new_capacity)
            capacity = new_capacity
            
        sequence.push_back(b)
        a, b = b, a + b

    # Convert to Python list efficiently using a memory view
    return [sequence[i] for i in range(sequence.size())]

