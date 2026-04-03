# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Sort integers using C stdlib qsort with function pointer callback (Cython-optimized).

Keywords: sorting, qsort, stdlib, callback, comparison, cython, benchmark
"""

from libc.stdlib cimport malloc, free, qsort
from cnake_charmer.benchmarks import cython_benchmark


cdef int _compare_ints(const void *a, const void *b) noexcept nogil:
    """Comparison callback for qsort: ascending order."""
    cdef int va = (<int *>a)[0]
    cdef int vb = (<int *>b)[0]
    if va < vb:
        return -1
    elif va > vb:
        return 1
    return 0


@cython_benchmark(syntax="cy", args=(50000,))
def stdlib_qsort(int n):
    """Sort n deterministic integers using C stdlib qsort and return checksum.

    Args:
        n: Number of integers to sort.

    Returns:
        Tuple of (sum_first_10, sum_last_10, total_sum).
    """
    cdef int *arr = <int *>malloc(n * sizeof(int))
    if not arr:
        raise MemoryError()

    cdef int i
    cdef unsigned long long v
    cdef int limit
    cdef long long sum_first = 0
    cdef long long sum_last = 0
    cdef long long total = 0

    with nogil:
        for i in range(n):
            v = ((<unsigned long long>i * <unsigned long long>2654435761) ^ (<unsigned long long>i * <unsigned long long>40503)) % <unsigned long long>1000000
            arr[i] = <int>v

        qsort(<void *>arr, <size_t>n, sizeof(int), _compare_ints)

        limit = 10 if n >= 10 else n
        for i in range(limit):
            sum_first += arr[i]

        for i in range(n - limit, n):
            sum_last += arr[i]

        for i in range(n):
            total += arr[i]

    free(arr)
    return (sum_first, sum_last, total)
