# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Compute sum of elements of the covariance matrix for 3 variables (Cython-optimized).

Variables: x1[i]=sin(i*0.1), x2[i]=cos(i*0.2), x3[i]=sin(i*0.3) with n observations.
Returns the sum of all 9 elements of the 3x3 covariance matrix.

Keywords: statistics, covariance, matrix, multivariate, cython, benchmark
"""

from libc.math cimport sin, cos
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(500000,))
def covariance_matrix(int n):
    """Compute sum of elements of the 3x3 covariance matrix."""
    cdef int i, a, b
    cdef double sum1, sum2, sum3, mean1, mean2, mean3
    cdef double v1, v2, v3, total
    cdef double cov[3][3]

    # Compute means
    sum1 = 0.0
    sum2 = 0.0
    sum3 = 0.0
    for i in range(n):
        sum1 += sin(i * 0.1)
        sum2 += cos(i * 0.2)
        sum3 += sin(i * 0.3)

    mean1 = sum1 / n
    mean2 = sum2 / n
    mean3 = sum3 / n

    # Zero the covariance matrix
    for a in range(3):
        for b in range(3):
            cov[a][b] = 0.0

    # Accumulate
    for i in range(n):
        v1 = sin(i * 0.1) - mean1
        v2 = cos(i * 0.2) - mean2
        v3 = sin(i * 0.3) - mean3
        cov[0][0] += v1 * v1
        cov[0][1] += v1 * v2
        cov[0][2] += v1 * v3
        cov[1][0] += v2 * v1
        cov[1][1] += v2 * v2
        cov[1][2] += v2 * v3
        cov[2][0] += v3 * v1
        cov[2][1] += v3 * v2
        cov[2][2] += v3 * v3

    total = 0.0
    for a in range(3):
        for b in range(3):
            total += cov[a][b] / (n - 1)

    return total
