"""Matrix chain multiplication with split tracking and DP diagnostics.

Keywords: dynamic programming, matrix chain, optimization, split, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(200,))
def matrix_chain_order(n: int) -> tuple:
    """Find minimum cost matrix chain multiplication with split point tracking.

    Matrix dimensions: d[i] = 15 + (i*31 + 17) % 85 for i in range(n+1).
    Uses classic O(n^3) DP with split point tracking.
    Returns (min_cost, first_split_point, dp_value_at_midpoint).

    Args:
        n: Number of matrices in the chain.

    Returns:
        Tuple of (min_cost, split_point_first, dp_mid_val).
    """
    if n < 2:
        return (0, 0, 0)

    # Generate dimensions
    d = [0] * (n + 1)
    for i in range(n + 1):
        d[i] = 15 + (i * 31 + 17) % 85

    # dp[i*n + j] = min cost to multiply matrices i..j
    dp = [0] * (n * n)
    split = [0] * (n * n)

    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            dp[i * n + j] = 2**62
            for k in range(i, j):
                cost = dp[i * n + k] + dp[(k + 1) * n + j] + d[i] * d[k + 1] * d[j + 1]
                if cost < dp[i * n + j]:
                    dp[i * n + j] = cost
                    split[i * n + j] = k

    min_cost = dp[0 * n + (n - 1)]
    split_first = split[0 * n + (n - 1)]

    # DP value at midpoint
    mid = n // 2
    dp_mid = dp[0 * n + mid] if mid > 0 and mid < n else 0

    return (min_cost, split_first, dp_mid)
