# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Quicksort with median-of-three pivot (Cython-optimized).

Keywords: sorting, quicksort, median of three, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_charmer.benchmarks import cython_benchmark


cdef int _median_of_three(int *arr, int lo, int hi):
    cdef int mid = (lo + hi) / 2
    cdef int a = arr[lo], b = arr[mid], c = arr[hi]
    if a <= b:
        if b <= c:
            return mid
        elif a <= c:
            return hi
        else:
            return lo
    else:
        if a <= c:
            return lo
        elif b <= c:
            return hi
        else:
            return mid


cdef void _insertion_sort(int *arr, int lo, int hi):
    cdef int i, j, temp
    for i in range(lo + 1, hi + 1):
        temp = arr[i]
        j = i
        while j > lo and arr[j - 1] > temp:
            arr[j] = arr[j - 1]
            j -= 1
        arr[j] = temp


cdef void _quicksort_impl(int *arr, int lo_init, int hi_init):
    # Iterative quicksort using explicit stack
    cdef int *stack_lo = <int *>malloc(64 * sizeof(int))
    cdef int *stack_hi = <int *>malloc(64 * sizeof(int))
    if not stack_lo or not stack_hi:
        if stack_lo:
            free(stack_lo)
        if stack_hi:
            free(stack_hi)
        return

    cdef int sp = 0
    cdef int lo, hi, pivot_idx, pivot, i, j, temp

    stack_lo[0] = lo_init
    stack_hi[0] = hi_init
    sp = 1

    while sp > 0:
        sp -= 1
        lo = stack_lo[sp]
        hi = stack_hi[sp]

        if lo >= hi:
            continue

        if hi - lo < 16:
            _insertion_sort(arr, lo, hi)
            continue

        pivot_idx = _median_of_three(arr, lo, hi)
        # Swap pivot to end
        temp = arr[pivot_idx]
        arr[pivot_idx] = arr[hi]
        arr[hi] = temp
        pivot = arr[hi]

        i = lo
        for j in range(lo, hi):
            if arr[j] <= pivot:
                temp = arr[i]
                arr[i] = arr[j]
                arr[j] = temp
                i += 1
        temp = arr[i]
        arr[i] = arr[hi]
        arr[hi] = temp

        stack_lo[sp] = lo
        stack_hi[sp] = i - 1
        sp += 1
        stack_lo[sp] = i + 1
        stack_hi[sp] = hi
        sp += 1

    free(stack_lo)
    free(stack_hi)


@cython_benchmark(syntax="cy", args=(200000,))
def quick_sort(int n):
    """Sort a deterministic array using quicksort with median-of-three pivot."""
    cdef int *arr = <int *>malloc(n * sizeof(int))
    if not arr:
        raise MemoryError()

    cdef int i
    for i in range(n):
        arr[i] = (i * 31 + 17) % n

    _quicksort_impl(arr, 0, n - 1)

    cdef list result = [arr[i] for i in range(n)]
    free(arr)
    return result
