# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Levenberg-Marquardt fitting of sinusoidal model (Cython-optimized).

Keywords: levenberg-marquardt, curve fitting, sinusoidal, optimization, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from libc.math cimport sin, cos, fabs
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(50000,))
def levenberg_marquardt(int n):
    """Fit y = a*sin(b*x + c) to n data points using simplified LM."""
    cdef double *x_data = <double *>malloc(n * sizeof(double))
    cdef double *y_data = <double *>malloc(n * sizeof(double))
    if not x_data or not y_data:
        if x_data:
            free(x_data)
        if y_data:
            free(y_data)
        raise MemoryError()

    cdef int i, it
    cdef double xi, yi, arg, sin_val, cos_val, pred, ri
    cdef double j0, j1, j2, noise
    cdef double a, b, c, lam
    cdef double jtj00, jtj01, jtj02, jtj11, jtj12, jtj22
    cdef double jtr0, jtr1, jtr2
    cdef double det, inv_det, dp0, dp1, dp2
    cdef double rss, diff
    # Temp for 3x3 matrix elements
    cdef double m00, m01, m02, m10, m11, m12, m20, m21, m22

    # Generate data
    for i in range(n):
        x_data[i] = i * 0.01
        noise = 0.05 * ((i * 13 + 7) % 100 - 50) / 50.0
        y_data[i] = 3.0 * sin(2.0 * x_data[i] + 1.0) + noise

    a = 2.0
    b = 1.5
    c = 0.5
    lam = 0.01

    for it in range(5):
        jtj00 = 0.0
        jtj01 = 0.0
        jtj02 = 0.0
        jtj11 = 0.0
        jtj12 = 0.0
        jtj22 = 0.0
        jtr0 = 0.0
        jtr1 = 0.0
        jtr2 = 0.0

        for i in range(n):
            xi = x_data[i]
            yi = y_data[i]
            arg = b * xi + c
            sin_val = sin(arg)
            cos_val = cos(arg)
            pred = a * sin_val
            ri = yi - pred

            j0 = sin_val
            j1 = a * xi * cos_val
            j2 = a * cos_val

            jtr0 += j0 * ri
            jtr1 += j1 * ri
            jtr2 += j2 * ri

            jtj00 += j0 * j0
            jtj01 += j0 * j1
            jtj02 += j0 * j2
            jtj11 += j1 * j1
            jtj12 += j1 * j2
            jtj22 += j2 * j2

        # Add damping and solve
        m00 = jtj00 + lam
        m01 = jtj01
        m02 = jtj02
        m10 = jtj01
        m11 = jtj11 + lam
        m12 = jtj12
        m20 = jtj02
        m21 = jtj12
        m22 = jtj22 + lam

        det = (m00 * (m11 * m22 - m12 * m21)
               - m01 * (m10 * m22 - m12 * m20)
               + m02 * (m10 * m21 - m11 * m20))

        if fabs(det) < 1e-30:
            break

        inv_det = 1.0 / det
        dp0 = inv_det * (jtr0 * (m11 * m22 - m12 * m21)
                         - jtr1 * (m01 * m22 - m02 * m21)
                         + jtr2 * (m01 * m12 - m02 * m11))
        dp1 = inv_det * (m00 * (jtr1 * m22 - jtr2 * m21)
                         - jtr0 * (m10 * m22 - m12 * m20)
                         + m02 * (jtr2 * m10 - jtr1 * m20))
        dp2 = inv_det * (m00 * (m11 * jtr2 - jtr1 * m21)
                         - m01 * (m10 * jtr2 - jtr1 * m20)
                         + jtr0 * (m10 * m21 - m11 * m20))

        a += dp0
        b += dp1
        c += dp2

    rss = 0.0
    for i in range(n):
        pred = a * sin(b * x_data[i] + c)
        diff = y_data[i] - pred
        rss += diff * diff

    free(x_data)
    free(y_data)

    return rss
