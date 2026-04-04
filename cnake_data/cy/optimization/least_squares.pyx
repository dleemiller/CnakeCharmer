# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Fit exponential model using Gauss-Newton least squares (Cython-optimized).

Keywords: least squares, gauss-newton, exponential fit, optimization, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from libc.math cimport exp, fabs
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(10000,))
def least_squares(int n):
    """Fit y = a*exp(b*x) to n data points using Gauss-Newton iteration."""
    cdef double *x_data = <double *>malloc(n * sizeof(double))
    cdef double *y_data = <double *>malloc(n * sizeof(double))
    if not x_data or not y_data:
        if x_data:
            free(x_data)
        if y_data:
            free(y_data)
        raise MemoryError()

    cdef int i, it
    cdef double xi, yi, eb, pred, ri, j0, j1, bxi
    cdef double jtj00, jtj01, jtj11, jtr0, jtr1
    cdef double det, da, db
    cdef double a, b

    # Generate data
    for i in range(n):
        x_data[i] = i * 0.01
        y_data[i] = 2.5 * exp(0.3 * x_data[i]) + 0.1 * ((i * 7 + 3) % 100 - 50) / 50.0

    # Initial guess
    a = 1.0
    b = 0.1

    # Gauss-Newton: 10 iterations
    for it in range(10):
        jtj00 = 0.0
        jtj01 = 0.0
        jtj11 = 0.0
        jtr0 = 0.0
        jtr1 = 0.0

        for i in range(n):
            xi = x_data[i]
            yi = y_data[i]
            bxi = b * xi
            if bxi > 500.0:
                bxi = 500.0
            elif bxi < -500.0:
                bxi = -500.0
            eb = exp(bxi)
            pred = a * eb
            ri = yi - pred
            j0 = eb
            j1 = a * xi * eb

            jtj00 += j0 * j0
            jtj01 += j0 * j1
            jtj11 += j1 * j1
            jtr0 += j0 * ri
            jtr1 += j1 * ri

        det = jtj00 * jtj11 - jtj01 * jtj01
        if fabs(det) < 1e-30:
            break
        da = (jtj11 * jtr0 - jtj01 * jtr1) / det
        db = (jtj00 * jtr1 - jtj01 * jtr0) / det

        a += da
        b += db

    free(x_data)
    free(y_data)

    return a + b
