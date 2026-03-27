# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Newton's method square root with except? error return spec (Cython-optimized).

Keywords: numerical, square root, newton, error handling, except spec, cython, benchmark
"""

from cnake_charmer.benchmarks import cython_benchmark


cdef double _safe_sqrt(double x) except? -1.0:
    """Compute square root using Newton's method.

    Returns -1.0 for negative input. Uses except? -1.0 so Cython
    checks PyErr_Occurred only when -1.0 is returned.
    """
    if x < 0.0:
        return -1.0
    if x == 0.0:
        return 0.0
    cdef double guess = x * 0.5
    cdef int i
    for i in range(30):
        guess = 0.5 * (guess + x / guess)
    return guess


@cython_benchmark(syntax="cy", args=(100000,))
def except_check_sqrt(int n):
    """Compute square roots using cdef with except? -1.0 spec."""
    cdef double total = 0.0
    cdef int count = 0
    cdef int i
    cdef int val
    cdef double result

    for i in range(n):
        val = ((i * 67 + 23) % 2003) - 1000
        result = _safe_sqrt(<double>val)
        if result >= 0.0:
            total += result
            count += 1

    return total + <double>count
