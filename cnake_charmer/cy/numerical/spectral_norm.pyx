# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Spectral norm of a matrix via power iteration.

Keywords: spectral norm, eigenvalue, power iteration, linear algebra, matrix, cython
"""

from libc.math cimport sqrt
from libc.stdlib cimport malloc, free

from cnake_charmer.benchmarks import cython_benchmark


cdef inline double A_elem(int i, int j) nogil:
    return 1.0 / ((i + j) * (i + j + 1) / 2 + i + 1)


cdef void A_times_u(double *u, double *v, int n) nogil:
    cdef int i, j
    cdef double s
    for i in range(n):
        s = 0.0
        for j in range(n):
            s += A_elem(i, j) * u[j]
        v[i] = s


cdef void At_times_u(double *u, double *v, int n) nogil:
    cdef int i, j
    cdef double s
    for i in range(n):
        s = 0.0
        for j in range(n):
            s += A_elem(j, i) * u[j]
        v[i] = s


@cython_benchmark(syntax="cy", args=(300,))
def spectral_norm(int n):
    """Compute spectral norm of the n×n truncation of matrix A.

    Args:
        n: Matrix dimension.

    Returns:
        Tuple of (norm_value, u_checksum, v_checksum).
    """
    cdef double *u = <double *>malloc(n * sizeof(double))
    cdef double *v = <double *>malloc(n * sizeof(double))
    cdef double *tmp = <double *>malloc(n * sizeof(double))
    if not u or not v or not tmp:
        raise MemoryError()

    cdef int i
    for i in range(n):
        u[i] = 1.0
        v[i] = 1.0
        tmp[i] = 0.0

    for _ in range(10):
        # v = A^T * A * u
        A_times_u(u, tmp, n)
        At_times_u(tmp, v, n)
        # u = A^T * A * v
        A_times_u(v, tmp, n)
        At_times_u(tmp, u, n)

    cdef double vbv = 0.0
    cdef double vv = 0.0
    for i in range(n):
        vbv += u[i] * v[i]
        vv += v[i] * v[i]

    cdef double norm_val = sqrt(vbv / vv)

    cdef double u_check = 0.0
    cdef double v_check = 0.0
    cdef int kk = 10 if n >= 10 else n
    for i in range(kk):
        u_check += u[i]
        v_check += v[i]

    free(u)
    free(v)
    free(tmp)
    return (norm_val, u_check, v_check)
