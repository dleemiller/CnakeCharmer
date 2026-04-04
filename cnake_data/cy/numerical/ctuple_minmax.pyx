# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Find min and max of hash-derived values and return their
sum (Cython-optimized with ctuple return).

Keywords: ctuple, min, max, numerical, cython, benchmark
"""

from libc.stdlib cimport malloc, free

from cnake_data.benchmarks import cython_benchmark


cdef (double, double) find_minmax(double *arr, int n):
    """Find min and max in array, return as ctuple."""
    cdef double mn = arr[0]
    cdef double mx = arr[0]
    cdef int i
    for i in range(1, n):
        if arr[i] < mn:
            mn = arr[i]
        if arr[i] > mx:
            mx = arr[i]
    return (mn, mx)


@cython_benchmark(syntax="cy", args=(100000,))
def ctuple_minmax(int n):
    """Compute min and max of n hash-derived values."""
    cdef double *arr = <double *>malloc(
        n * sizeof(double)
    )
    if not arr:
        raise MemoryError()
    cdef int i
    cdef unsigned long long h
    for i in range(n):
        h = (<unsigned long long>i
             * <unsigned long long>2654435761)
        arr[i] = (
            <double>(h & <unsigned long long>0xFFFFFFFF)
            / 4294967295.0 * 200.0 - 100.0
        )
    cdef (double, double) result = find_minmax(arr, n)
    free(arr)
    return result[0] + result[1]
