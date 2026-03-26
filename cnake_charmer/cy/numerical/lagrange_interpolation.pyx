# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Lagrange polynomial interpolation (Cython-optimized).

Keywords: numerical, interpolation, lagrange, polynomial, cython, benchmark
"""

from libc.math cimport sin, M_PI
from libc.stdlib cimport malloc, free
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(5000,))
def lagrange_interpolation(int n):
    """Interpolate at x=0.5 using n Lagrange basis polynomials with C arrays."""
    if n < 2:
        return 0.0

    cdef double *xs = <double *>malloc(n * sizeof(double))
    cdef double *ys = <double *>malloc(n * sizeof(double))
    if not xs or not ys:
        if xs: free(xs)
        if ys: free(ys)
        raise MemoryError()

    cdef int i, j
    cdef double target = 0.5
    cdef double result = 0.0
    cdef double basis, nm1

    nm1 = <double>(n - 1)
    for i in range(n):
        xs[i] = i / nm1
        ys[i] = sin(xs[i] * M_PI)

    for i in range(n):
        basis = 1.0
        for j in range(n):
            if j != i:
                basis *= (target - xs[j]) / (xs[i] - xs[j])
        result += ys[i] * basis

    free(xs)
    free(ys)
    return result
