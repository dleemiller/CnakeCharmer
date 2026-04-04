# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Batch bisection root finding (Cython-optimized).

Keywords: bisection, root finding, batch, trigonometric, optimization, cython, benchmark
"""

from libc.math cimport sin, M_PI
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(200000,))
def bisection_batch(int n):
    """Find roots of f(x) = sin(x) - x/k for n values of k."""
    cdef double pi = M_PI
    cdef double total = 0.0
    cdef int i, j
    cdef double k, lo, hi, mid, fmid

    with nogil:
        for i in range(n):
            k = <double>(i + 1)
            lo = 0.0
            hi = pi

            for j in range(50):
                mid = 0.5 * (lo + hi)
                fmid = sin(mid) - mid / k
                if fmid > 0.0:
                    lo = mid
                else:
                    hi = mid

            total += 0.5 * (lo + hi)

    return total
