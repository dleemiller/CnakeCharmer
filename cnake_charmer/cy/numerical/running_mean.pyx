# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""
Compute running mean of a deterministically generated sequence (Cython-optimized).

Keywords: running mean, cumulative average, numerical, statistics, cython, benchmark
"""

from cnake_charmer.benchmarks import cython_benchmark
import cython


@cython_benchmark(syntax="cy", args=(100000,))
def running_mean(int n):
    """Compute the running mean using C-typed cumulative sum."""
    cdef list result = []
    cdef double cumsum = 0.0
    cdef double value
    cdef int i

    for i in range(n):
        value = ((i * 7 + 3) % 1000) / 10.0
        cumsum += value
        result.append(cumsum / (i + 1))

    return result
