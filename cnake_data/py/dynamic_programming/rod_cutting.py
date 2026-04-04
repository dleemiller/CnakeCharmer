"""
Maximum revenue from cutting a rod of length n.

Keywords: dynamic programming, rod cutting, optimization, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(3000,))
def rod_cutting(n: int) -> int:
    """Compute maximum revenue from cutting a rod of length n.

    Prices: price[i] = (i*7+3) % 50 + 1 for piece of length i (1..n).
    Classic bottom-up DP.

    Args:
        n: Rod length.

    Returns:
        Maximum revenue as an integer.
    """
    price = [0] * (n + 1)
    for i in range(1, n + 1):
        price[i] = (i * 7 + 3) % 50 + 1

    dp = [0] * (n + 1)

    for j in range(1, n + 1):
        best = 0
        for i in range(1, j + 1):
            val = price[i] + dp[j - i]
            if val > best:
                best = val
        dp[j] = best

    return dp[n]
