# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""
Count inversions in a deterministic permutation using merge sort (Cython-optimized).

Keywords: algorithms, inversions, merge sort, counting, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from libc.string cimport memcpy
from cnake_data.benchmarks import cython_benchmark


cdef long long merge_count_impl(int *arr, int *temp, int left, int right):
    """Merge sort with inversion counting on C arrays."""
    cdef long long inversions = 0
    cdef int mid, i, j, k

    if right - left <= 1:
        return 0

    mid = (left + right) / 2
    inversions += merge_count_impl(arr, temp, left, mid)
    inversions += merge_count_impl(arr, temp, mid, right)

    i = left
    j = mid
    k = left
    while i < mid and j < right:
        if arr[i] <= arr[j]:
            temp[k] = arr[i]
            i += 1
        else:
            temp[k] = arr[j]
            inversions += mid - i
            j += 1
        k += 1

    while i < mid:
        temp[k] = arr[i]
        i += 1
        k += 1

    while j < right:
        temp[k] = arr[j]
        j += 1
        k += 1

    memcpy(&arr[left], &temp[left], (right - left) * sizeof(int))
    return inversions


@cython_benchmark(syntax="cy", args=(50000,))
def count_inversions(int n):
    """Count inversions in a deterministic permutation using merge sort on C arrays."""
    cdef int *arr = <int *>malloc(n * sizeof(int))
    cdef int *temp = <int *>malloc(n * sizeof(int))

    if arr == NULL or temp == NULL:
        if arr != NULL:
            free(arr)
        if temp != NULL:
            free(temp)
        raise MemoryError("Failed to allocate arrays")

    cdef int i
    for i in range(n):
        arr[i] = (i * 7 + 13) % n

    cdef long long result = merge_count_impl(arr, temp, 0, n)

    # Compute checksum from sorted array (merge sort sorted it in-place)
    cdef long long checksum = 0
    cdef long long MOD = 1000000007
    for i in range(n):
        checksum = (checksum + <long long>i * <long long>arr[i]) % MOD

    free(arr)
    free(temp)
    return (result, int(checksum))
