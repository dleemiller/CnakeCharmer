# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Find min and max of int and double arrays using fused types.

Keywords: numerical, min, max, array, fused type, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_data.benchmarks import cython_benchmark

ctypedef fused numeric_t:
    int
    double


cdef void _minmax_int(int *arr, int n, int *out_min, int *out_max) noexcept:
    cdef int i
    cdef int mn = arr[0]
    cdef int mx = arr[0]
    for i in range(1, n):
        if arr[i] < mn:
            mn = arr[i]
        if arr[i] > mx:
            mx = arr[i]
    out_min[0] = mn
    out_max[0] = mx


cdef void _minmax_double(double *arr, int n, double *out_min, double *out_max) noexcept:
    cdef int i
    cdef double mn = arr[0]
    cdef double mx = arr[0]
    for i in range(1, n):
        if arr[i] < mn:
            mn = arr[i]
        if arr[i] > mx:
            mx = arr[i]
    out_min[0] = mn
    out_max[0] = mx


@cython_benchmark(syntax="cy", args=(100000,))
def fused_minmax(int n):
    """Find min/max of int and double arrays using fused type helpers."""
    cdef int *int_arr = <int *>malloc(n * sizeof(int))
    cdef double *dbl_arr = <double *>malloc(n * sizeof(double))
    if not int_arr or not dbl_arr:
        raise MemoryError()

    cdef int i
    for i in range(n):
        int_arr[i] = (i * 31 + 17) % 100003

    for i in range(n):
        dbl_arr[i] = ((i * 37 + 11) % 99991) / 7.0

    cdef int min_int, max_int
    cdef double min_dbl, max_dbl

    _minmax_int(int_arr, n, &min_int, &max_int)
    _minmax_double(dbl_arr, n, &min_dbl, &max_dbl)

    free(int_arr)
    free(dbl_arr)
    return <double>min_int + <double>max_int + min_dbl + max_dbl
