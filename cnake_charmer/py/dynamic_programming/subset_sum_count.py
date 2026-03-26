"""
Count subsets that sum to a target value.

Keywords: dynamic programming, subset sum, counting, modular arithmetic, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(500,))
def subset_sum_count(n: int) -> int:
    """Count subsets of n items that sum to target = n * 3.

    Items: v[i] = (i*7+3) % 20 + 1 for i in 0..n-1.
    Result is computed mod 10^9 + 7.

    Args:
        n: Number of items.

    Returns:
        Count of subsets mod 10^9+7.
    """
    MOD = 1000000007
    target = n * 3
    items = [(i * 7 + 3) % 20 + 1 for i in range(n)]

    # dp[j] = number of subsets summing to j
    dp = [0] * (target + 1)
    dp[0] = 1

    for i in range(n):
        v = items[i]
        for j in range(target, v - 1, -1):
            dp[j] = (dp[j] + dp[j - v]) % MOD

    return dp[target]
