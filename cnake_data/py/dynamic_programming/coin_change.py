"""
Minimum coin change problem.

Keywords: dynamic programming, coin change, optimization, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(50000,))
def coin_change(n: int) -> int:
    """Find minimum number of coins to make amount n.

    Uses denominations [1, 5, 10, 25].

    Args:
        n: Target amount.

    Returns:
        Minimum number of coins needed.
    """
    coins = [1, 5, 10, 25]
    dp = [0] * (n + 1)

    for i in range(1, n + 1):
        dp[i] = dp[i - 1] + 1  # Using coin of value 1
        for c in coins[1:]:
            if c <= i and dp[i - c] + 1 < dp[i]:
                dp[i] = dp[i - c] + 1

    return (dp[n], dp[n // 2])
