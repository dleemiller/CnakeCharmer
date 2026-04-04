# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""
Compute prefix maximum of a deterministically generated sequence (Cython-optimized).

Keywords: numerical, prefix maximum, running max, cython, benchmark
"""

from cpython.array cimport array, clone
from array import array as py_array
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(500000,))
def prefix_max(int n):
    """Compute the prefix maximum using a C-typed memoryview for zero-overhead writes."""
    cdef int i
    cdef int value
    cdef int current_max = 0
    cdef array template = py_array('i', [])
    cdef array buf = clone(template, n, zero=False)
    cdef int[:] view = buf

    for i in range(n):
        value = (i * 31 + 17) % 10000
        if value > current_max:
            current_max = value
        view[i] = current_max

    return buf.tolist()
