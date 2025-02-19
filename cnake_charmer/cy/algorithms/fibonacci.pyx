# cython: boundscheck=False, wraparound=False, language_level=3
"""
Fibonacci Module
----------------

This module implements an optimized Fibonacci function that computes all Fibonacci numbers
below a given value using a preallocated Python array and memory views for efficient C-level access.

The implementation uses Cython 3 pure Python syntax with type annotations and Google-style docstrings.
Long integers are used to handle large Fibonacci numbers.
"""

import cython
from cython.cimports.cpython import array as carray
import array as pyarray
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(1e18,))
def fib(n: cython.longlong) -> list[cython.longlong]:
    """Compute Fibonacci numbers less than n using a preallocated array and memory views.

    Args:
        n (cython.longlong): The upper limit for the Fibonacci sequence (exclusive).

    Returns:
        list[cython.longlong]: A list of Fibonacci numbers that are less than n.

    This function preallocates a Python array of long integers using the Cython clone function for efficient memory
    management. It then obtains a memory view to allow fast, C-level access while computing the Fibonacci numbers.
    If the preallocated space is exceeded, the array is resized smartly. Finally, the computed sequence is converted
    into a Python list for returning.
    """
    long_array_template: pyarray.array = pyarray.array('q', [])  # 'q' for long long
    capacity: cython.int = 100
    # Preallocate the array using Cython's clone function from carray.
    fib_array: pyarray.array = carray.clone(long_array_template, capacity, zero=False)

    idx: cython.int = 0
    a: cython.longlong = 0
    b: cython.longlong = 1
    fib_view: cython.longlong[:] = fib_array  # Obtain a memory view for fast C-level access.

    while b < n:
        # If the preallocated array is full, extend its capacity.
        if idx >= len(fib_view):
            new_capacity = len(fib_view) + (len(fib_view) // 2)
            pyarray.resize(fib_array, new_capacity)
            fib_view = fib_array  # Update the memory view after resizing.
        fib_view[idx] = b
        idx += 1
        a, b = b, a + b

    # Convert the memory view slice to a Python list.
    return [fib_view[i] for i in range(idx)]

