# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Linear least squares via normal equations (Cython-optimized).

Keywords: linear least squares, normal equations, polynomial fit, optimization, cython, benchmark
"""

from libc.math cimport sin, fabs
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(1000000,))
def linear_least_squares(int n):
    """Solve overdetermined Ax=b in least squares sense."""
    cdef int i
    cdef double t, t2, bi
    cdef double ata00, ata01, ata02, ata11, ata12, ata22
    cdef double atb0, atb1, atb2
    cdef double det, inv_det, x0, x1, x2

    ata00 = 0.0
    ata01 = 0.0
    ata02 = 0.0
    ata11 = 0.0
    ata12 = 0.0
    ata22 = 0.0
    atb0 = 0.0
    atb1 = 0.0
    atb2 = 0.0

    for i in range(n):
        t = i * 0.01
        t2 = t * t
        bi = sin(t)

        ata00 += 1.0
        ata01 += t
        ata02 += t2
        ata11 += t2
        ata12 += t * t2
        ata22 += t2 * t2

        atb0 += bi
        atb1 += t * bi
        atb2 += t2 * bi

    # Solve 3x3 via Cramer's rule (symmetric: ata10=ata01, etc.)
    det = (ata00 * (ata11 * ata22 - ata12 * ata12)
           - ata01 * (ata01 * ata22 - ata12 * ata02)
           + ata02 * (ata01 * ata12 - ata11 * ata02))

    if fabs(det) < 1e-30:
        return 0.0

    inv_det = 1.0 / det

    x0 = inv_det * (atb0 * (ata11 * ata22 - ata12 * ata12)
                    - atb1 * (ata01 * ata22 - ata02 * ata12)
                    + atb2 * (ata01 * ata12 - ata02 * ata11))
    x1 = inv_det * (ata00 * (atb1 * ata22 - atb2 * ata12)
                    - atb0 * (ata01 * ata22 - ata12 * ata02)
                    + ata02 * (atb2 * ata01 - atb1 * ata02))
    x2 = inv_det * (ata00 * (ata11 * atb2 - atb1 * ata12)
                    - ata01 * (ata01 * atb2 - atb1 * ata02)
                    + atb0 * (ata01 * ata12 - ata11 * ata02))

    return x0 + x1 + x2
