"""Pancake sort using prefix reversals.

Keywords: sorting, pancake sort, prefix reversal, flips, algorithm, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(3000,))
def pancake_sort(n: int) -> tuple:
    """Sort a deterministic array using pancake sort (prefix reversals only).

    Generates arr[i] = (i * 2654435761) % n, then sorts by repeatedly
    finding the max element and flipping prefixes.

    Args:
        n: Number of elements to sort.

    Returns:
        Tuple of (checksum of sorted array, number of flips performed).
    """
    arr = [((i * 2654435761) & 0xFFFFFFFF) % n for i in range(n)]

    flips = 0

    for size in range(n, 1, -1):
        # Find index of maximum element in arr[0..size-1]
        max_idx = 0
        for i in range(1, size):
            if arr[i] > arr[max_idx]:
                max_idx = i

        if max_idx == size - 1:
            continue

        # Flip to bring max to front
        if max_idx > 0:
            lo, hi = 0, max_idx
            while lo < hi:
                arr[lo], arr[hi] = arr[hi], arr[lo]
                lo += 1
                hi -= 1
            flips += 1

        # Flip to put max at end of current range
        lo, hi = 0, size - 1
        while lo < hi:
            arr[lo], arr[hi] = arr[hi], arr[lo]
            lo += 1
            hi -= 1
        flips += 1

    # Compute checksum: weighted sum of sorted positions
    checksum = 0
    for i in range(n):
        checksum += arr[i] * (i + 1)

    return (checksum, flips)
