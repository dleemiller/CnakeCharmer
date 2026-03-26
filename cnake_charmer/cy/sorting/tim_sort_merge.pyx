# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Simulate the merge phase of Tim sort (Cython-optimized).

Keywords: tim sort, merge, sorting, runs, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from libc.string cimport memcpy
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(500000,))
def tim_sort_merge(int n):
    """Sort array using Tim sort merge strategy with C arrays."""
    cdef int *arr = <int *>malloc(n * sizeof(int))
    cdef int *tmp = <int *>malloc(n * sizeof(int))
    if not arr or not tmp:
        free(arr)
        free(tmp)
        raise MemoryError()

    cdef int i, j, start, end, key, size, left, mid, right
    cdef int ii, jj, kk
    cdef int run_size = 32

    # Generate array
    for i in range(n):
        arr[i] = (i * 31 + 17) % n

    # Insertion sort each run
    for start in range(0, n, run_size):
        end = start + run_size
        if end > n:
            end = n
        for i in range(start + 1, end):
            key = arr[i]
            j = i - 1
            while j >= start and arr[j] > key:
                arr[j + 1] = arr[j]
                j -= 1
            arr[j + 1] = key

    # Merge runs pairwise
    size = run_size
    while size < n:
        for left in range(0, n, 2 * size):
            mid = left + size
            if mid > n:
                mid = n
            right = left + 2 * size
            if right > n:
                right = n
            if mid < right:
                # Merge arr[left:mid] and arr[mid:right] into tmp
                ii = left
                jj = mid
                kk = left
                while ii < mid and jj < right:
                    if arr[ii] <= arr[jj]:
                        tmp[kk] = arr[ii]
                        ii += 1
                    else:
                        tmp[kk] = arr[jj]
                        jj += 1
                    kk += 1
                while ii < mid:
                    tmp[kk] = arr[ii]
                    ii += 1
                    kk += 1
                while jj < right:
                    tmp[kk] = arr[jj]
                    jj += 1
                    kk += 1
                memcpy(&arr[left], &tmp[left], (right - left) * sizeof(int))
        size *= 2

    cdef list result = [arr[i] for i in range(n)]
    free(arr)
    free(tmp)
    return result
