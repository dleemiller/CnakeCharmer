# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Sum arrays using fused type helpers for int and double, return combined sum.

Keywords: numerical, array, sum, fused type, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_charmer.benchmarks import cython_benchmark

ctypedef fused numeric_t:
    int
    long long
    double


cdef double _sum_array_int(int *arr, int n) noexcept:
    cdef int i
    cdef long long total = 0
    for i in range(n):
        total += arr[i]
    return <double>total


cdef double _sum_array_double(double *arr, int n) noexcept:
    cdef int i
    cdef double total = 0.0
    for i in range(n):
        total += arr[i]
    return total


@cython_benchmark(syntax="cy", args=(100000,))
def fused_array_sum(int n):
    """Sum an int array and a double array using fused type helpers."""
    cdef int *int_arr = <int *>malloc(n * sizeof(int))
    cdef double *dbl_arr = <double *>malloc(n * sizeof(double))
    if not int_arr or not dbl_arr:
        raise MemoryError()

    cdef int i
    for i in range(n):
        int_arr[i] = (i * 7 + 3) % 1000

    for i in range(n):
        dbl_arr[i] = ((i * 11 + 5) % 1000) / 10.0

    cdef double int_sum = _sum_array_int(int_arr, n)
    cdef double dbl_sum = _sum_array_double(dbl_arr, n)

    free(int_arr)
    free(dbl_arr)
    return int_sum + dbl_sum
