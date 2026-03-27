"""Binary search with error handling on a sorted array.

Keywords: algorithms, binary search, error handling, except, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


def _binary_search(arr, n, target):
    """Binary search returning index or negative value if not found."""
    lo = 0
    hi = n - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -(n + 1)


@python_benchmark(args=(100000,))
def except_value_search(n: int) -> int:
    """Perform n binary searches on a sorted array, count found items.

    Array: sorted values arr[i] = i * 3 (so not all integers present).
    Search targets: target = (i * 2654435761) % (n * 4) for each i.
    Count how many searches succeed (return >= 0).

    Args:
        n: Number of searches to perform (also array size).

    Returns:
        Number of successful searches.
    """
    arr = [0] * n
    for i in range(n):
        arr[i] = i * 3

    found_count = 0
    mask = 0xFFFFFFFF
    for i in range(n):
        target = ((i * 2654435761) & mask) % (n * 4)
        idx = _binary_search(arr, n, target)
        if idx >= 0:
            found_count += 1

    return found_count
