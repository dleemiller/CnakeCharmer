# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Introsort: quicksort with heapsort fallback (Cython-optimized).

Keywords: sorting, introsort, quicksort, heapsort, hybrid, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_charmer.benchmarks import cython_benchmark


cdef void _sift_down(int *arr, int start, int end):
    cdef int root, child, temp
    root = start
    while True:
        child = 2 * root - start + 1
        if child > end:
            break
        if child + 1 <= end and arr[child] < arr[child + 1]:
            child += 1
        if arr[root] < arr[child]:
            temp = arr[root]
            arr[root] = arr[child]
            arr[child] = temp
            root = child
        else:
            break


cdef void _heapsort(int *arr, int lo, int hi):
    cdef int count, start, end, temp
    count = hi - lo + 1
    for start in range((count - 2) / 2 + lo, lo - 1, -1):
        _sift_down(arr, start, hi)
    end = hi
    while end > lo:
        temp = arr[lo]
        arr[lo] = arr[end]
        arr[end] = temp
        end -= 1
        _sift_down(arr, lo, end)


cdef void _insertion_sort(int *arr, int lo, int hi):
    cdef int i, j, temp
    for i in range(lo + 1, hi + 1):
        temp = arr[i]
        j = i
        while j > lo and arr[j - 1] > temp:
            arr[j] = arr[j - 1]
            j -= 1
        arr[j] = temp


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


cdef void _introsort_impl(int *arr, int lo_init, int hi_init, int depth_limit):
    cdef int *stack_lo = <int *>malloc(128 * sizeof(int))
    cdef int *stack_hi = <int *>malloc(128 * sizeof(int))
    cdef int *stack_depth = <int *>malloc(128 * sizeof(int))
    if not stack_lo or not stack_hi or not stack_depth:
        if stack_lo:
            free(stack_lo)
        if stack_hi:
            free(stack_hi)
        if stack_depth:
            free(stack_depth)
        return

    cdef int sp = 0, lo, hi, depth, size, pivot_idx, pivot, store, j, temp

    stack_lo[0] = lo_init
    stack_hi[0] = hi_init
    stack_depth[0] = depth_limit
    sp = 1

    while sp > 0:
        sp -= 1
        lo = stack_lo[sp]
        hi = stack_hi[sp]
        depth = stack_depth[sp]
        size = hi - lo + 1

        if size <= 1:
            continue
        if size < 16:
            _insertion_sort(arr, lo, hi)
            continue
        if depth == 0:
            _heapsort(arr, lo, hi)
            continue

        pivot_idx = _median_of_three(arr, lo, hi)
        temp = arr[pivot_idx]
        arr[pivot_idx] = arr[hi]
        arr[hi] = temp
        pivot = arr[hi]

        store = lo
        for j in range(lo, hi):
            if arr[j] <= pivot:
                temp = arr[store]
                arr[store] = arr[j]
                arr[j] = temp
                store += 1
        temp = arr[store]
        arr[store] = arr[hi]
        arr[hi] = temp

        stack_lo[sp] = lo
        stack_hi[sp] = store - 1
        stack_depth[sp] = depth - 1
        sp += 1
        stack_lo[sp] = store + 1
        stack_hi[sp] = hi
        stack_depth[sp] = depth - 1
        sp += 1

    free(stack_lo)
    free(stack_hi)
    free(stack_depth)


@cython_benchmark(syntax="cy", args=(200000,))
def introsort(int n):
    """Sort a deterministic array using introsort."""
    cdef int *arr = <int *>malloc(n * sizeof(int))
    if not arr:
        raise MemoryError()

    cdef int i, depth_limit, tmp
    for i in range(n):
        arr[i] = (i * 31 + 17) % n

    depth_limit = 0
    tmp = n
    while tmp > 0:
        depth_limit += 1
        tmp >>= 1
    depth_limit *= 2

    _introsort_impl(arr, 0, n - 1, depth_limit)

    cdef list result = [arr[i] for i in range(n)]
    free(arr)
    return result
