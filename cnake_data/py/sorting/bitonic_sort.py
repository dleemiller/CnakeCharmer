"""Bitonic sort network for power-of-2 arrays.

Keywords: sorting, bitonic sort, sorting network, parallel sort, algorithm, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(32768,))
def bitonic_sort(n: int) -> tuple:
    """Sort a deterministic array using bitonic sort network.

    Generates arr[i] = (i * 2654435761) % n for a power-of-2 sized array,
    then sorts using the bitonic merge sort network.

    Args:
        n: Number of elements (must be a power of 2).

    Returns:
        Tuple of (checksum of sorted array, number of compare-swaps performed).
    """
    arr = [((i * 2654435761) & 0xFFFFFFFF) % n for i in range(n)]

    comparisons = 0

    # Bitonic sort: build increasingly large bitonic sequences then merge
    k = 2
    while k <= n:
        j = k >> 1
        while j > 0:
            for i in range(n):
                partner = i ^ j
                if partner > i:
                    # Determine sort direction: ascending if (i & k) == 0
                    if (i & k) == 0:
                        if arr[i] > arr[partner]:
                            arr[i], arr[partner] = arr[partner], arr[i]
                    else:
                        if arr[i] < arr[partner]:
                            arr[i], arr[partner] = arr[partner], arr[i]
                    comparisons += 1
            j >>= 1
        k <<= 1

    # Compute checksum: weighted sum of sorted positions
    checksum = 0
    for i in range(n):
        checksum += arr[i] * (i + 1)

    return (checksum, comparisons)
