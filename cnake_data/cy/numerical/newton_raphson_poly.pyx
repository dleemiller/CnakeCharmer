# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Newton-Raphson root finding for polynomial x^3 - 2x - 1 (Cython-optimized).

Keywords: numerical, newton-raphson, root finding, polynomial, fractal, cython, benchmark
"""

from libc.math cimport fabs

from cnake_data.benchmarks import cython_benchmark


cdef inline double poly_f(double x) noexcept nogil:
    return x * x * x - 2.0 * x - 1.0


cdef inline double poly_fp(double x) noexcept nogil:
    return 3.0 * x * x - 2.0


@cython_benchmark(syntax="cy", args=(50000,))
def newton_raphson_poly(int n):
    """Find roots of x^3 - 2x - 1 using Newton-Raphson from many starting points."""
    cdef double checksum = 0.0
    cdef int converge_count = 0
    cdef double step, x, x_new, fx, fpx, fx_final
    cdef int i, j

    if n > 1:
        step = 6.0 / (n - 1)
    else:
        step = 0.0

    with nogil:
        for i in range(n):
            x = -3.0 + i * step

            for j in range(50):
                fx = poly_f(x)
                fpx = poly_fp(x)
                if fpx == 0.0:
                    break
                x_new = x - fx / fpx
                if fabs(x_new - x) < 1e-10:
                    x = x_new
                    break
                x = x_new

            checksum += x

            fx_final = poly_f(x)
            if fabs(fx_final) < 1e-8:
                converge_count += 1

    return (checksum, converge_count)
