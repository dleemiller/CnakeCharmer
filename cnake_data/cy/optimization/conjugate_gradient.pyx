# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Conjugate gradient solver for tridiagonal system (Cython-optimized).

Keywords: conjugate gradient, linear solver, tridiagonal, optimization, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from libc.math cimport fabs
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(5000,))
def conjugate_gradient(int n):
    """Solve Ax=b with conjugate gradient on a tridiagonal matrix."""
    cdef double *x = <double *>malloc(n * sizeof(double))
    cdef double *r = <double *>malloc(n * sizeof(double))
    cdef double *p = <double *>malloc(n * sizeof(double))
    cdef double *ap = <double *>malloc(n * sizeof(double))

    if not x or not r or not p or not ap:
        if x:
            free(x)
        if r:
            free(r)
        if p:
            free(p)
        if ap:
            free(ap)
        raise MemoryError()

    cdef int i, it
    cdef int max_iter = 2 * n
    cdef double rs_old, rs_new, pap, alpha, beta, total

    # Initialize
    for i in range(n):
        x[i] = 0.0
        r[i] = 1.0  # b - A*x = b since x=0
        p[i] = 1.0

    rs_old = 0.0
    for i in range(n):
        rs_old += r[i] * r[i]

    for it in range(max_iter):
        # Compute A*p (tridiagonal: -1, 4, -1)
        for i in range(n):
            ap[i] = 4.0 * p[i]
            if i > 0:
                ap[i] -= p[i - 1]
            if i < n - 1:
                ap[i] -= p[i + 1]

        pap = 0.0
        for i in range(n):
            pap += p[i] * ap[i]

        if fabs(pap) < 1e-30:
            break

        alpha = rs_old / pap

        rs_new = 0.0
        for i in range(n):
            x[i] += alpha * p[i]
            r[i] -= alpha * ap[i]
            rs_new += r[i] * r[i]

        if rs_new < 1e-20:
            break

        beta = rs_new / rs_old

        for i in range(n):
            p[i] = r[i] + beta * p[i]

        rs_old = rs_new

    total = 0.0
    for i in range(n):
        total += x[i]

    free(x)
    free(r)
    free(p)
    free(ap)

    return total
