# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Sort array using function pointer typedef for comparison (Cython-optimized).

Keywords: sorting, callback, function pointer, ctypedef, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_charmer.benchmarks import cython_benchmark

ctypedef int (*compare_fn)(int, int) noexcept


cdef int _compare_asc(int a, int b) noexcept:
    if a < b:
        return -1
    elif a > b:
        return 1
    return 0


cdef int _compare_desc(int a, int b) noexcept:
    if a > b:
        return -1
    elif a < b:
        return 1
    return 0


cdef void _insertion_sort(int *arr, int n, compare_fn cmp) noexcept:
    """Insertion sort using a function pointer comparator."""
    cdef int i, j, key
    for i in range(1, n):
        key = arr[i]
        j = i - 1
        while j >= 0 and cmp(arr[j], key) > 0:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key


cdef void _qsort_helper(int *arr, int lo, int hi, compare_fn cmp) noexcept:
    """Quicksort using function pointer comparator."""
    if lo >= hi:
        return
    cdef int pivot = arr[(lo + hi) / 2]
    cdef int i = lo
    cdef int j = hi
    cdef int tmp
    while i <= j:
        while cmp(arr[i], pivot) < 0:
            i += 1
        while cmp(arr[j], pivot) > 0:
            j -= 1
        if i <= j:
            tmp = arr[i]
            arr[i] = arr[j]
            arr[j] = tmp
            i += 1
            j -= 1
    _qsort_helper(arr, lo, j, cmp)
    _qsort_helper(arr, i, hi, cmp)


@cython_benchmark(syntax="cy", args=(50000,))
def typedef_callback_sort(int n):
    """Sort array using function pointer typedef comparators."""
    cdef int *arr = <int *>malloc(n * sizeof(int))
    if not arr:
        raise MemoryError()

    cdef int i
    cdef long long val
    for i in range(n):
        val = (<long long>i * <long long>1103515245 + <long long>12345) & <long long>0x7FFFFFFF
        arr[i] = <int>(val % <long long>1000000)

    # Ascending sort
    _qsort_helper(arr, 0, n - 1, _compare_asc)
    cdef long long checksum = 0
    cdef int limit = 10 if n >= 10 else n
    for i in range(limit):
        checksum += arr[i]

    # Descending sort
    _qsort_helper(arr, 0, n - 1, _compare_desc)
    for i in range(limit):
        checksum += arr[i]

    free(arr)
    return <int>checksum
