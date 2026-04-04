"""
Minimum cost path in a grid from top-left to bottom-right.

Keywords: dynamic programming, minimum cost, grid path, optimization, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(1200,))
def min_cost_path(n: int) -> tuple:
    """Find minimum cost path in an n x n grid.

    Cost of cell (i,j) = ((i * 31 + j * 37 + 5) % 100) + 1, range [1, 100].
    Can move right or down only.

    Args:
        n: Grid dimension (n x n).

    Returns:
        Tuple of (min_cost_to_bottom_right, min_cost_to_center).
    """
    # Build cost grid inline and compute DP
    dp = [[0] * n for _ in range(n)]
    dp[0][0] = ((0 * 31 + 0 * 37 + 5) % 100) + 1

    # First row
    for j in range(1, n):
        cost = ((0 * 31 + j * 37 + 5) % 100) + 1
        dp[0][j] = dp[0][j - 1] + cost

    # First column
    for i in range(1, n):
        cost = ((i * 31 + 0 * 37 + 5) % 100) + 1
        dp[i][0] = dp[i - 1][0] + cost

    # Fill table
    for i in range(1, n):
        for j in range(1, n):
            cost = ((i * 31 + j * 37 + 5) % 100) + 1
            if dp[i - 1][j] < dp[i][j - 1]:
                dp[i][j] = dp[i - 1][j] + cost
            else:
                dp[i][j] = dp[i][j - 1] + cost

    mid = n // 2
    return (dp[n - 1][n - 1], dp[mid][mid])
