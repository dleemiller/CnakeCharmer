# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Patience sort on a deterministic integer array (Cython-optimized).

Keywords: algorithms, patience sort, sorting, piles, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(100000,))
def patience_sort(int n):
    """Sort array using patience sort with C arrays for piles and heap merge."""
    cdef int *arr = <int *>malloc(n * sizeof(int))
    cdef int *output = <int *>malloc(n * sizeof(int))
    if not arr or not output:
        if arr: free(arr)
        if output: free(output)
        raise MemoryError()

    cdef int i, val, lo_idx, hi_idx, mid_idx

    # Generate input
    for i in range(n):
        arr[i] = (i * 31 + 17) % n

    # Use Python lists for piles since pile count is dynamic
    # But typed loop variables for the hot path
    piles = []
    cdef int num_piles = 0

    # pile_tops tracks the top of each pile for binary search
    cdef int *pile_tops = <int *>malloc(n * sizeof(int))
    if not pile_tops:
        free(arr)
        free(output)
        raise MemoryError()

    for i in range(n):
        val = arr[i]
        # Binary search for leftmost pile whose top >= val
        lo_idx = 0
        hi_idx = num_piles
        while lo_idx < hi_idx:
            mid_idx = (lo_idx + hi_idx) / 2
            if pile_tops[mid_idx] >= val:
                hi_idx = mid_idx
            else:
                lo_idx = mid_idx + 1
        if lo_idx == num_piles:
            piles.append([val])
            pile_tops[num_piles] = val
            num_piles += 1
        else:
            piles[lo_idx].append(val)
            pile_tops[lo_idx] = val

    free(pile_tops)
    free(arr)

    # Merge using Python heapq (pile count is small relative to n)
    import heapq
    heap = []
    cdef int idx
    for idx in range(num_piles):
        heap.append((piles[idx].pop(), idx))
    heapq.heapify(heap)

    cdef int out_idx = 0
    while heap:
        val, idx = heapq.heappop(heap)
        output[out_idx] = val
        out_idx += 1
        if piles[idx]:
            heapq.heappush(heap, (piles[idx].pop(), idx))

    cdef list result = [output[i] for i in range(n)]
    free(output)
    return result
