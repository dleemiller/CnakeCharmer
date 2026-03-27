# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Clamp array values using fused type helpers.

Keywords: numerical, clamp, array, fused type, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_charmer.benchmarks import cython_benchmark

ctypedef fused numeric_t:
    int
    double


cdef long long _clamp_sum_int(int *arr, int n, int lo, int hi) noexcept:
    cdef int i
    cdef int val
    cdef long long total = 0
    for i in range(n):
        val = arr[i]
        if val < lo:
            val = lo
        elif val > hi:
            val = hi
        total += val
    return total


cdef double _clamp_sum_double(double *arr, int n, double lo, double hi) noexcept:
    cdef int i
    cdef double val
    cdef double total = 0.0
    for i in range(n):
        val = arr[i]
        if val < lo:
            val = lo
        elif val > hi:
            val = hi
        total += val
    return total


@cython_benchmark(syntax="cy", args=(100000,))
def fused_clamp(int n):
    """Clamp int and double arrays using fused type helpers."""
    cdef int *int_arr = <int *>malloc(n * sizeof(int))
    cdef double *dbl_arr = <double *>malloc(n * sizeof(double))
    if not int_arr or not dbl_arr:
        raise MemoryError()

    cdef int i
    for i in range(n):
        int_arr[i] = (i * 47 + 13) % 2003 - 1000

    for i in range(n):
        dbl_arr[i] = ((i * 53 + 7) % 1999 - 999) / 3.0

    cdef double int_sum = <double>_clamp_sum_int(int_arr, n, -200, 200)
    cdef double dbl_sum = _clamp_sum_double(dbl_arr, n, -50.0, 50.0)

    free(int_arr)
    free(dbl_arr)
    return int_sum + dbl_sum
