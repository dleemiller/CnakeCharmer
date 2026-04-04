# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Accumulation loop with division (Cython-optimized).

Keywords: accumulation, loop, division, numerical, benchmark, cython
"""

from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(10000000,))
def accumulate_divisions(int n):
    """Accumulate sum of i/200 for i=1..n."""
    cdef double y = 1.0
    cdef double i_d
    cdef int i
    for i in range(1, n + 1):
        i_d = <double>i
        y += i_d / 200.0
    return y
