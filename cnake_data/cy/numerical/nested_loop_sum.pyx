# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Nested loop accumulation (Cython-optimized).

Keywords: nested_loop, accumulation, integer, multiplication, numerical, cython
"""

from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(5000,))
def nested_loop_sum(int n):
    """Compute sum of i*j*n over nested loops."""
    cdef long long total = 0
    cdef int i, j
    for i in range(n):
        for j in range(100):
            total += i * j * n
    return total % (2 ** 63)
