"""Maximum subarray sum using Kadane's algorithm.

Keywords: leetcode, max subarray, kadane, dynamic programming, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(5000000,))
def max_subarray(n: int) -> int:
    """Find the maximum subarray sum using Kadane's algorithm.

    Array is generated as v[i] = ((i * 17 + 5) % 201) - 100.

    Args:
        n: Size of the array.

    Returns:
        Maximum subarray sum.
    """
    best = current = ((0 * 17 + 5) % 201) - 100
    for i in range(1, n):
        val = ((i * 17 + 5) % 201) - 100
        current = current + val if current + val > val else val
        if current > best:
            best = current
    return best
