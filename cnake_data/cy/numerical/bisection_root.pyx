# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Generic bisection root finder using function pointers (Cython-optimized).

Keywords: numerical, bisection, root finding, function pointer, cython, benchmark
"""

from cnake_data.benchmarks import cython_benchmark

ctypedef double (*math_fn)(double) noexcept


cdef double _f1(double x) noexcept:
    return x * x - 2.0


cdef double _f2(double x) noexcept:
    return x * x * x - x - 1.0


cdef double _f3(double x) noexcept:
    return x * x - 4.0 * x + 3.5


cdef double _f4(double x) noexcept:
    return x * x * x - 6.0 * x * x + 11.0 * x - 5.5


cdef double _bisect(math_fn func, double a, double b, int max_iter) noexcept:
    """Bisection method using function pointer."""
    cdef int i
    cdef double mid
    for i in range(max_iter):
        mid = (a + b) * 0.5
        if func(mid) * func(a) <= 0.0:
            b = mid
        else:
            a = mid
    return (a + b) * 0.5


@cython_benchmark(syntax="cy", args=(10000,))
def bisection_root(int n):
    """Find roots of 4 functions using bisection with function pointers."""
    cdef math_fn funcs[4]
    funcs[0] = _f1
    funcs[1] = _f2
    funcs[2] = _f3
    funcs[3] = _f4

    cdef double intervals_a[4]
    cdef double intervals_b[4]
    intervals_a[0] = 1.0; intervals_b[0] = 2.0
    intervals_a[1] = 1.0; intervals_b[1] = 2.0
    intervals_a[2] = 1.0; intervals_b[2] = 2.0
    intervals_a[3] = 0.5; intervals_b[3] = 1.5

    cdef double total = 0.0
    cdef int i, k

    for i in range(n):
        for k in range(4):
            total += _bisect(funcs[k], intervals_a[k], intervals_b[k], 50)

    return total
