# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Integer square root by Newton iteration (Cython)."""

from cnake_data.benchmarks import cython_benchmark


cdef inline int _isqrt(int k) noexcept:
    cdef int x = k
    cdef int y = (x + 1) // 2
    while y < x:
        x = y
        y = (x + k // x) // 2
    return x


@cython_benchmark(syntax="cy", args=(1, 7000))
def integer_sqrt_newton(int start, int stop):
    return _integer_sqrt_newton_impl(start, stop)


cdef int _integer_sqrt_newton_impl(int start, int stop) noexcept:
    cdef int total = 0
    cdef int k
    if start < 1:
        start = 1
    for k in range(start, stop):
        total += _isqrt(k)
    return total
