"""Gnome sort algorithm with swap counting.

Keywords: sorting, gnome sort, swap count, simple sort, algorithm, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(3000,))
def gnome_sort(n: int) -> tuple:
    """Sort a deterministic array using gnome sort, counting swaps.

    Generates arr[i] = (i * 2654435761) % n, then sorts using gnome sort.
    The gnome walks forward when elements are in order, and backward
    (swapping) when they are not.

    Args:
        n: Number of elements to sort.

    Returns:
        Tuple of (checksum of sorted array, number of swaps performed).
    """
    arr = [((i * 2654435761) & 0xFFFFFFFF) % n for i in range(n)]

    swaps = 0
    pos = 0

    while pos < n:
        if pos == 0 or arr[pos] >= arr[pos - 1]:
            pos += 1
        else:
            arr[pos], arr[pos - 1] = arr[pos - 1], arr[pos]
            swaps += 1
            pos -= 1

    # Compute checksum: weighted sum of sorted positions
    checksum = 0
    for i in range(n):
        checksum += arr[i] * (i + 1)

    return (checksum, swaps)
