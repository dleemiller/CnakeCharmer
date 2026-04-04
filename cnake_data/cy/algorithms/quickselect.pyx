# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Find kth smallest element using quickselect with deterministic pivoting.

Keywords: algorithms, quickselect, selection, partition, kth smallest, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(500000,))
def quickselect(int n):
    """Find multiple kth-smallest elements using quickselect partition scheme."""
    cdef int *arr = <int *>malloc(n * sizeof(int))
    cdef int *buf = <int *>malloc(n * sizeof(int))
    if not arr or not buf:
        if arr: free(arr)
        if buf: free(buf)
        raise MemoryError()

    cdef int i, lo, hi, k, mid, pivot_val, pivot_idx, store
    cdef int r1, r2, r3

    # Generate base array
    for i in range(n):
        arr[i] = (i * 37 + 13) % (n * 2)

    # Find three order statistics on separate copies
    cdef int targets[3]
    cdef int results[3]
    targets[0] = n // 4
    targets[1] = n // 2
    targets[2] = (3 * n) // 4

    cdef int t
    for t in range(3):
        # Copy arr into buf
        for i in range(n):
            buf[i] = arr[i]

        lo = 0
        hi = n - 1
        k = targets[t]

        while lo < hi:
            mid = (lo + hi) / 2
            # Median-of-three
            if buf[lo] > buf[mid]:
                buf[lo], buf[mid] = buf[mid], buf[lo]
            if buf[lo] > buf[hi]:
                buf[lo], buf[hi] = buf[hi], buf[lo]
            if buf[mid] > buf[hi]:
                buf[mid], buf[hi] = buf[hi], buf[mid]

            # Partition around mid
            pivot_val = buf[mid]
            buf[mid], buf[hi] = buf[hi], buf[mid]
            store = lo
            for i in range(lo, hi):
                if buf[i] < pivot_val:
                    buf[store], buf[i] = buf[i], buf[store]
                    store += 1
            buf[store], buf[hi] = buf[hi], buf[store]
            pivot_idx = store

            if pivot_idx == k:
                break
            elif pivot_idx < k:
                lo = pivot_idx + 1
            else:
                hi = pivot_idx - 1

        results[t] = buf[k]

    r1 = results[0]
    r2 = results[1]
    r3 = results[2]

    free(arr)
    free(buf)
    return (r1, r2, r3)
