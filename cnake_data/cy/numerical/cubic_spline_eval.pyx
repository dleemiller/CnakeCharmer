# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Evaluate a natural cubic spline at n midpoints (Cython-optimized).

Knots at x=0,1,...,n with y[i]=(i*7+3)%100. Evaluates the spline at the
midpoint of each interval and returns the sum of interpolated values.

Keywords: numerical, interpolation, cubic spline, tridiagonal, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(50000,))
def cubic_spline_eval(int n):
    """Evaluate natural cubic spline at n midpoints and return their sum."""
    cdef int i, m
    cdef double denom, t, mi, mi1, val, total
    cdef double *y = <double *>malloc((n + 1) * sizeof(double))
    cdef double *second_deriv = <double *>malloc((n + 1) * sizeof(double))
    cdef double *c_prime
    cdef double *d_prime
    cdef double *rhs

    if not y or not second_deriv:
        raise MemoryError()

    for i in range(n + 1):
        y[i] = <double>((i * 7 + 3) % 100)

    m = n - 1
    if m <= 0:
        val = y[0]
        free(y)
        free(second_deriv)
        return val

    rhs = <double *>malloc(m * sizeof(double))
    c_prime = <double *>malloc(m * sizeof(double))
    d_prime = <double *>malloc(m * sizeof(double))
    if not rhs or not c_prime or not d_prime:
        raise MemoryError()

    for i in range(m):
        rhs[i] = 6.0 * (y[i + 2] - 2.0 * y[i + 1] + y[i])

    # Forward elimination
    c_prime[0] = 1.0 / 4.0
    d_prime[0] = rhs[0] / 4.0

    for i in range(1, m):
        denom = 4.0 - c_prime[i - 1]
        c_prime[i] = 1.0 / denom
        d_prime[i] = (rhs[i] - d_prime[i - 1]) / denom

    # Natural spline: M[0] = M[n] = 0
    for i in range(n + 1):
        second_deriv[i] = 0.0

    second_deriv[m] = d_prime[m - 1]
    for i in range(m - 2, -1, -1):
        second_deriv[i + 1] = d_prime[i] - c_prime[i] * second_deriv[i + 2]

    # Evaluate at midpoints
    total = 0.0
    for i in range(n):
        t = 0.5
        mi = second_deriv[i]
        mi1 = second_deriv[i + 1]
        val = (mi * (1.0 - t) * (1.0 - t) * (1.0 - t) + mi1 * t * t * t) / 6.0
        val += (y[i] - mi / 6.0) * (1.0 - t) + (y[i + 1] - mi1 / 6.0) * t
        total += val

    free(y)
    free(second_deriv)
    free(rhs)
    free(c_prime)
    free(d_prime)
    return total
