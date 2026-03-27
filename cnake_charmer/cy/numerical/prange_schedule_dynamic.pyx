# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Variable-iteration convergence with dynamic schedule.

Keywords: numerical, Newton's method, prange, dynamic, parallel, cython, benchmark
"""

from cython.parallel import prange
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(500000,))
def prange_schedule_dynamic(int n):
    """Compute cube roots via Newton iterations, return sum."""
    cdef double total = 0.0
    cdef double val, x, x3
    cdef int i, _iter

    for i in prange(
        1, n + 1,
        nogil=True,
        schedule='dynamic',
        chunksize=64,
    ):
        val = <double>i
        x = val / 3.0
        for _iter in range(50):
            x3 = x * x * x
            if x3 - val < 1e-12 and val - x3 < 1e-12:
                break
            x = x - (x3 - val) / (3.0 * x * x)
        total += x

    return total
