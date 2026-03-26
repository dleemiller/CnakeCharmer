"""
Check if an array can be partitioned into two subsets with equal sum.

Keywords: dynamic programming, partition, subset sum, bitset, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(2000,))
def partition_equal_sum(n: int) -> int:
    """Check if array can be partitioned into two equal-sum subsets.

    Values: v[i] = (i*7+3) % 20 + 1 for i in 0..n-1.
    Uses boolean DP array where dp[j] indicates whether sum j is reachable.

    Args:
        n: Number of items.

    Returns:
        1 if partition exists, 0 otherwise.
    """
    total = 0
    items = [0] * n
    for i in range(n):
        items[i] = (i * 7 + 3) % 20 + 1
        total += items[i]

    # If total is odd, no equal partition possible
    if total % 2 != 0:
        return (0, 0)

    target = total // 2

    # Boolean DP: dp[j] = True if sum j is reachable
    dp = [False] * (target + 1)
    dp[0] = True

    for i in range(n):
        val = items[i]
        for j in range(target, val - 1, -1):
            if dp[j - val]:
                dp[j] = True

    reachable_count = sum(1 for x in dp if x)
    return (1 if dp[target] else 0, reachable_count)
