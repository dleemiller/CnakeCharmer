# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Analytic eigenvalue computation for 3x3 symmetric matrices (Cython-optimized).

Uses the trigonometric method to solve the characteristic cubic polynomial
for matrices of the form [[a,d,0],[d,b,e],[0,e,c]].

Keywords: numerical, eigenvalues, symmetric matrix, trigonometric method, cython, benchmark
"""

from cnake_data.benchmarks import cython_benchmark

from libc.math cimport cos, acos, sqrt

cdef double M_PI = 3.14159265358979323846
cdef double TWO_PI_OVER_3 = 2.0 * M_PI / 3.0


cdef (double, double, double) _eigenvalues_symmetric_3x3(
    double a, double b, double c, double d, double e
) noexcept:
    """Compute sorted eigenvalues of [[a,d,0],[d,b,e],[0,e,c]]."""
    cdef double p1, q, aq, bq, cq, p2, p, inv_p
    cdef double b00, b11, b22, b01, b12, det_b, half_det, phi
    cdef double lam1, lam2, lam3, tmp

    p1 = d * d + e * e
    q = (a + b + c) / 3.0

    aq = a - q
    bq = b - q
    cq = c - q

    p2 = aq * aq + bq * bq + cq * cq + 2.0 * p1
    p = sqrt(p2 / 6.0)

    if p < 1e-15:
        # Nearly diagonal / proportional to identity
        lam1 = a
        lam2 = b
        lam3 = c
        if lam1 > lam2:
            tmp = lam1; lam1 = lam2; lam2 = tmp
        if lam2 > lam3:
            tmp = lam2; lam2 = lam3; lam3 = tmp
        if lam1 > lam2:
            tmp = lam1; lam1 = lam2; lam2 = tmp
        return (lam1, lam2, lam3)

    inv_p = 1.0 / p

    b00 = aq * inv_p
    b11 = bq * inv_p
    b22 = cq * inv_p
    b01 = d * inv_p
    b12 = e * inv_p

    det_b = b00 * (b11 * b22 - b12 * b12) - b01 * (b01 * b22)

    half_det = det_b * 0.5
    if half_det <= -1.0:
        half_det = -1.0
    elif half_det >= 1.0:
        half_det = 1.0

    phi = acos(half_det) / 3.0

    lam1 = q + 2.0 * p * cos(phi)
    lam3 = q + 2.0 * p * cos(phi + TWO_PI_OVER_3)
    lam2 = 3.0 * q - lam1 - lam3

    if lam1 > lam2:
        tmp = lam1; lam1 = lam2; lam2 = tmp
    if lam2 > lam3:
        tmp = lam2; lam2 = lam3; lam3 = tmp
    if lam1 > lam2:
        tmp = lam1; lam1 = lam2; lam2 = tmp

    return (lam1, lam2, lam3)


@cython_benchmark(syntax="cy", args=(50000,))
def eigenvalues_3x3(int n):
    """Compute eigenvalues of n deterministic symmetric 3x3 matrices."""
    cdef double sum_min = 0.0
    cdef double sum_max = 0.0
    cdef double trace_check = 0.0
    cdef double a, b, c, d, e
    cdef double lam1, lam2, lam3
    cdef int i

    for i in range(n):
        a = i * 0.1 + 1.0
        b = i * 0.2 + 2.0
        c = i * 0.15 + 1.5
        d = i * 0.05 + 0.3
        e = i * 0.03 + 0.2

        lam1, lam2, lam3 = _eigenvalues_symmetric_3x3(a, b, c, d, e)

        sum_min += lam1
        sum_max += lam3
        trace_check += (lam1 + lam2 + lam3) - (a + b + c)

    return (sum_min, sum_max, trace_check)
