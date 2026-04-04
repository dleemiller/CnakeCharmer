# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Compute square roots via Newton's method with GIL release.

Keywords: numerical, Newton's method, square root, nogil, cython, benchmark
"""

from cnake_data.benchmarks import cython_benchmark


cdef double _newton_sqrt(double x) noexcept nogil:
    """Compute sqrt(x) via 10 Newton iterations."""
    cdef double guess = x * 0.5
    cdef int i
    for i in range(10):
        guess = 0.5 * (guess + x / guess)
    return guess


@cython_benchmark(syntax="cy", args=(500000,))
def nogil_newton_roots(int n):
    """Compute sqrt of n values via Newton's method."""
    cdef double total = 0.0
    cdef int i

    with nogil:
        for i in range(1, n + 1):
            total += _newton_sqrt(<double>i)

    return total
