"""Count elements found via binary search in a sorted array.

Keywords: sorted array, binary search, container, sequence protocol, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(100000,))
def sorted_array_search(n: int) -> int:
    """Build a sorted array of n elements, then search for n/2 random values.

    Returns the count of successful lookups.

    Args:
        n: Number of elements in the sorted array.

    Returns:
        Number of values found.
    """
    # Build sorted array with gaps
    arr = [0] * n
    for i in range(n):
        arr[i] = i * 3 + 1

    size = n
    found = 0
    for i in range(n // 2):
        # Generate search target
        target = ((i * 2654435761 + 17) >> 4) % (n * 4)

        # Binary search
        lo = 0
        hi = size - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            val = arr[mid]
            if val == target:
                found += 1
                break
            elif val < target:
                lo = mid + 1
            else:
                hi = mid - 1

    return found
