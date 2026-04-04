"""Count the number of ways to make change for amount n using coins [1, 2, 5, 10, 25].

Keywords: dynamic programming, coin change, counting, combinatorics, benchmark
"""

from cnake_data.benchmarks import python_benchmark

MOD = 10**9 + 7


@python_benchmark(args=(5000,))
def coin_change_count(n: int) -> tuple:
    """Count distinct ways to make change for amount n using coins [1, 2, 5, 10, 25].

    Uses 1D DP: dp[i] = number of ways to make amount i (mod 10^9+7).

    Args:
        n: Target amount.

    Returns:
        Tuple of (dp[n], dp[n//2], dp[n//4], dp[n//3]) all mod 10^9+7.
    """
    coins = [1, 2, 5, 10, 25]
    dp = [0] * (n + 1)
    dp[0] = 1

    for coin in coins:
        for i in range(coin, n + 1):
            dp[i] = (dp[i] + dp[i - coin]) % MOD

    return (dp[n], dp[n // 2], dp[n // 4], dp[n // 3])
