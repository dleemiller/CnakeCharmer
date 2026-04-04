"""Minimum coin change for amounts 1..n with total tracking.

Keywords: dynamic programming, coin change, minimum coins, optimization, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(200000,))
def coin_change_totals(n: int) -> tuple:
    """Compute minimum coins for each amount from 1 to n.

    Denominations: [1, 3, 7, 11, 23].
    Returns coins for n, coins for mid, and total coins across all amounts.

    Args:
        n: Maximum target amount.

    Returns:
        Tuple of (coins_for_n, coins_for_mid, total_coins).
    """
    coins = [1, 3, 7, 11, 23]
    dp = [0] * (n + 1)

    for i in range(1, n + 1):
        dp[i] = dp[i - 1] + 1  # Use coin of value 1
        for c in coins[1:]:
            if c <= i and dp[i - c] + 1 < dp[i]:
                dp[i] = dp[i - c] + 1

    mid = n // 2
    total_coins = 0
    for i in range(1, n + 1):
        total_coins += dp[i]

    return (dp[n], dp[mid], total_coins)
