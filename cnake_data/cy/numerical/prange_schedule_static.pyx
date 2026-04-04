# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Polynomial evaluation with uniform workload (static schedule).

Keywords: numerical, polynomial, prange, static, parallel, cython, benchmark
"""

from cython.parallel import prange
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(1000000,))
def prange_schedule_static(int n):
    """Evaluate degree-8 polynomial at n points, return sum."""
    cdef double total = 0.0
    cdef double x, val
    cdef int i

    for i in prange(n, nogil=True, schedule='static'):
        x = ((i * 13 + 7) % 10000) / 10000.0
        val = 8.0
        val = val * x + 7.0
        val = val * x + 6.0
        val = val * x + 5.0
        val = val * x + 4.0
        val = val * x + 3.0
        val = val * x + 2.0
        val = val * x + 1.0
        total += val

    return total
