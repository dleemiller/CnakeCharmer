"""Patience sort on a deterministic integer array.

Keywords: algorithms, patience sort, sorting, piles, benchmark
"""

import heapq

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(100000,))
def patience_sort(n: int) -> list:
    """Sort array using patience sort with piles and heap merge.

    Array: arr[i] = (i*31+17) % n.

    Args:
        n: Number of elements to sort.

    Returns:
        Sorted list of integers.
    """
    arr = [(i * 31 + 17) % n for i in range(n)]

    # Build piles (each pile is a list, we track the top)
    piles = []
    for val in arr:
        # Binary search for leftmost pile whose top >= val
        lo_idx = 0
        hi_idx = len(piles)
        while lo_idx < hi_idx:
            mid_idx = (lo_idx + hi_idx) // 2
            if piles[mid_idx][-1] >= val:
                hi_idx = mid_idx
            else:
                lo_idx = mid_idx + 1
        if lo_idx == len(piles):
            piles.append([val])
        else:
            piles[lo_idx].append(val)

    # Merge piles using a min-heap
    # Each pile has smallest on top (last element), pop from end
    heap = []
    for idx in range(len(piles)):
        heapq.heappush(heap, (piles[idx].pop(), idx))

    result = []
    for _ in range(n):
        val, idx = heapq.heappop(heap)
        result.append(val)
        if piles[idx]:
            heapq.heappush(heap, (piles[idx].pop(), idx))

    return result
