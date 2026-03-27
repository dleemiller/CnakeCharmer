"""Generic binary search on int and double sorted arrays.

Demonstrates fused type for generic search: same algorithm
works on both int and double arrays.

Keywords: algorithms, binary search, fused type, generic, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(100000,))
def fused_search(n: int) -> int:
    """Binary search on sorted int and double arrays.

    Creates sorted arrays and searches for n/3 target values
    in each. Returns total number of found targets.

    Args:
        n: Number of elements per array.

    Returns:
        Count of successful searches across both arrays.
    """
    # Build sorted int array
    int_arr = [(i * 3 + 1) for i in range(n)]

    # Build sorted double array
    dbl_arr = [(i * 2.7 + 0.5) for i in range(n)]

    found = 0
    num_queries = n // 3

    # Search int array
    for q in range(num_queries):
        target = (q * 7 + 3) % (n * 3)
        lo, hi = 0, n - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if int_arr[mid] < target:
                lo = mid + 1
            elif int_arr[mid] > target:
                hi = mid - 1
            else:
                found += 1
                break

    # Search double array
    for q in range(num_queries):
        target_d = (q * 11 + 5) % n * 2.7 + 0.5
        lo, hi = 0, n - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if dbl_arr[mid] < target_d - 1e-9:
                lo = mid + 1
            elif dbl_arr[mid] > target_d + 1e-9:
                hi = mid - 1
            else:
                found += 1
                break

    return found
