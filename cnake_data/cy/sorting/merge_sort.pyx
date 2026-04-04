# cython: boundscheck=False, wraparound=False, language_level=3
"""
Merge sort (Cython-optimized).

Keywords: sorting, merge sort, cython, benchmark
"""

from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(50000,))
def merge_sort(int n):
    """Sort a reversed list using typed merge sort."""
    cdef list arr = list(range(n, 0, -1))
    _merge_sort_impl(arr, 0, n - 1)
    return arr


cdef void _merge_sort_impl(list arr, int left, int right):
    cdef int mid
    if left < right:
        mid = (left + right) // 2
        _merge_sort_impl(arr, left, mid)
        _merge_sort_impl(arr, mid + 1, right)
        _merge_impl(arr, left, mid, right)


cdef void _merge_impl(list arr, int left, int mid, int right):
    cdef list left_half = arr[left:mid + 1]
    cdef list right_half = arr[mid + 1:right + 1]
    cdef int i = 0
    cdef int j = 0
    cdef int k = left
    cdef int left_len = mid + 1 - left
    cdef int right_len = right - mid

    while i < left_len and j < right_len:
        if <int>left_half[i] <= <int>right_half[j]:
            arr[k] = left_half[i]
            i += 1
        else:
            arr[k] = right_half[j]
            j += 1
        k += 1

    while i < left_len:
        arr[k] = left_half[i]
        i += 1
        k += 1

    while j < right_len:
        arr[k] = right_half[j]
        j += 1
        k += 1
