# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Compute abs/fabs sums using cdef extern from.

Keywords: numerical, extern, abs, fabs, cdef extern, cython, benchmark
"""

from cnake_data.benchmarks import cython_benchmark

cdef extern from "stdlib.h":
    int abs(int x) nogil

cdef extern from "math.h":
    double fabs(double x) nogil


@cython_benchmark(syntax="cy", args=(100000,))
def extern_abs_sum(int n):
    """Sum abs(int) and fabs(double) for n values."""
    cdef long long int_sum = 0
    cdef double float_sum = 0.0
    cdef int i, val
    cdef long long seed
    cdef double fval

    for i in range(n):
        val = (
            (<long long>i * <long long>2654435761 + 17)
            & 0x7FFFFFFF
        ) % 10000 - 5000
        int_sum += abs(val)

    for i in range(n):
        seed = (
            <long long>i * <long long>1103515245 + 12345
        ) & 0x7FFFFFFF
        fval = (seed % 100000) / 100.0 - 500.0
        float_sum += fabs(fval)

    return int_sum + float_sum
