"""
Binary search count algorithm.

Keywords: algorithms, binary search, searching, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(100000,))
def binary_search_count(n: int) -> int:
    """Count how many query values exist in a sorted array using binary search.

    Sorted array: arr[i] = i * 3 for i in range(n).
    Query array: query[j] = j * 5 for j in range(n).

    Args:
        n: Size of both arrays.

    Returns:
        Tuple of (count of found values, index of last found match or -1).
    """
    arr = [i * 3 for i in range(n)]
    count = 0
    last_found_idx = -1

    for j in range(n):
        target = j * 5
        lo = 0
        hi = n - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if arr[mid] == target:
                count += 1
                last_found_idx = mid
                break
            elif arr[mid] < target:
                lo = mid + 1
            else:
                hi = mid - 1

    return (count, last_found_idx)
