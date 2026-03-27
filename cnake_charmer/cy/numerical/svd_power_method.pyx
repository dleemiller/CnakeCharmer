# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""SVD largest singular value via power iteration (Cython-optimized).

Keywords: numerical, svd, singular value, power iteration, linear algebra, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from libc.math cimport sin, cos, sqrt, fabs
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(250,))
def svd_power_method(int n):
    """Compute largest singular value of n x n matrix via power iteration."""
    cdef double *A = <double *>malloc(n * n * sizeof(double))
    cdef double *v = <double *>malloc(n * sizeof(double))
    cdef double *w = <double *>malloc(n * sizeof(double))
    cdef double *u = <double *>malloc(n * sizeof(double))

    if not A or not v or not w or not u:
        free(A); free(v); free(w); free(u)
        raise MemoryError()

    cdef int i, j, it
    cdef double s, norm, sigma
    cdef int diff
    cdef int num_iters = 30

    # Build deterministic matrix
    for i in range(n):
        for j in range(n):
            diff = i - j
            if diff < 0:
                diff = -diff
            A[i * n + j] = sin((i + 1) * 0.1) * cos((j + 1) * 0.2) + 0.5 / (diff + 1)

    # Initialize v
    for i in range(n):
        v[i] = 0.0
    v[0] = 1.0

    sigma = 0.0

    for it in range(num_iters):
        # w = A * v
        for i in range(n):
            s = 0.0
            for j in range(n):
                s += A[i * n + j] * v[j]
            w[i] = s

        # u = A^T * w
        for i in range(n):
            s = 0.0
            for j in range(n):
                s += A[j * n + i] * w[j]
            u[i] = s

        # Compute norm
        norm = 0.0
        for i in range(n):
            norm += u[i] * u[i]
        norm = sqrt(norm)

        if norm < 1e-15:
            break

        # Normalize
        for i in range(n):
            v[i] = u[i] / norm

        sigma = sqrt(norm)

    cdef double v_first = v[0]
    cdef double v_last = v[n - 1]

    free(A)
    free(v)
    free(w)
    free(u)

    return (sigma, v_first, v_last)
