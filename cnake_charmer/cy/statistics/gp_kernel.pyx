# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Gaussian process squared exponential kernel matrix.

Keywords: gaussian process, kernel, covariance, rbf, squared exponential, cython
"""

from libc.math cimport exp
from libc.stdlib cimport malloc, free

from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(200,))
def gp_kernel(int n):
    """Compute GP covariance matrix for n data points in 5 dimensions.

    Args:
        n: Number of data points.

    Returns:
        Tuple of (trace, off_diag_sum, max_off_diag).
    """
    cdef int d = 5
    cdef double sigma = 1.0
    cdef double tau = 0.5
    cdef double epsilon = 1e-6
    cdef double inv_2tau2 = 1.0 / (2.0 * tau * tau)
    cdef int i, j, k
    cdef double dist_sq, diff, val
    cdef double trace = 0.0
    cdef double off_diag_sum = 0.0
    cdef double max_off_diag = 0.0

    # Allocate input features
    cdef double *X = <double *>malloc(n * d * sizeof(double))
    if not X:
        raise MemoryError()

    for i in range(n):
        for k in range(d):
            X[i * d + k] = ((i * 7 + k * 13 + 3) % 97) / 97.0

    with nogil:
        for i in range(n):
            trace += sigma + epsilon

            for j in range(i + 1, n):
                dist_sq = 0.0
                for k in range(d):
                    diff = X[i * d + k] - X[j * d + k]
                    dist_sq += diff * diff
                val = sigma * exp(-dist_sq * inv_2tau2)
                off_diag_sum += val
                if val > max_off_diag:
                    max_off_diag = val

    free(X)
    return (trace, off_diag_sum, max_off_diag)
