# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Prefix sum (accumulate) using fused type helpers.

Keywords: numerical, prefix sum, accumulate, fused type, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_charmer.benchmarks import cython_benchmark

ctypedef fused numeric_t:
    int
    double


cdef long long _accumulate_int(int *arr, int n) noexcept:
    """In-place prefix sum on int array, return last element."""
    cdef int i
    for i in range(1, n):
        arr[i] = arr[i] + arr[i - 1]
    return <long long>arr[n - 1]


cdef double _accumulate_double(double *arr, int n) noexcept:
    """In-place prefix sum on double array, return last element."""
    cdef int i
    for i in range(1, n):
        arr[i] = arr[i] + arr[i - 1]
    return arr[n - 1]


@cython_benchmark(syntax="cy", args=(100000,))
def fused_accumulate(int n):
    """Compute prefix sums using fused type helpers."""
    cdef int *int_arr = <int *>malloc(n * sizeof(int))
    cdef double *dbl_arr = <double *>malloc(n * sizeof(double))
    if not int_arr or not dbl_arr:
        raise MemoryError()

    cdef int i
    for i in range(n):
        int_arr[i] = (i * 23 + 5) % 509

    for i in range(n):
        dbl_arr[i] = ((i * 29 + 11) % 601) / 13.0

    cdef double int_result = <double>_accumulate_int(int_arr, n)
    cdef double dbl_result = _accumulate_double(dbl_arr, n)

    free(int_arr)
    free(dbl_arr)
    return int_result + dbl_result
