# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Power iteration for dominant eigenvalue (Cython-optimized).

Keywords: numerical, eigenvalue, power iteration, linear algebra, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from libc.math cimport fabs
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(300,))
def power_iteration(int n):
    """Find dominant eigenvalue of an n x n matrix using power iteration."""
    cdef double *M = <double *>malloc(n * n * sizeof(double))
    if not M:
        raise MemoryError()

    cdef double *v = <double *>malloc(n * sizeof(double))
    if not v:
        free(M)
        raise MemoryError()

    cdef double *w = <double *>malloc(n * sizeof(double))
    if not w:
        free(M)
        free(v)
        raise MemoryError()

    cdef int i, j, k
    cdef double s, max_val, eigenvalue

    # Build matrix
    for i in range(n):
        for j in range(n):
            M[i * n + j] = ((i * j + 3) % 10) / 10.0

    # Initial vector
    for i in range(n):
        v[i] = 1.0

    eigenvalue = 0.0

    for k in range(100):
        # Matrix-vector multiply
        for i in range(n):
            s = 0.0
            for j in range(n):
                s += M[i * n + j] * v[j]
            w[i] = s

        # Find max absolute value
        max_val = 0.0
        for i in range(n):
            if fabs(w[i]) > max_val:
                max_val = fabs(w[i])

        if max_val == 0.0:
            break

        eigenvalue = max_val

        # Normalize
        for i in range(n):
            v[i] = w[i] / max_val

    free(M)
    free(v)
    free(w)
    return eigenvalue
