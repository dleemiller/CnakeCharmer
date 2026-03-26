"""
Count unique paths in a grid with obstacles using DP.

Keywords: dynamic programming, grid paths, obstacles, combinatorics, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(800,))
def count_paths_grid(n: int) -> tuple:
    """Count unique paths in an n x n grid with deterministic obstacles.

    Cell (i,j) is blocked if ((i*7 + j*13 + 3) % 17) == 0.
    Paths go from top-left to bottom-right, moving only right or down.
    Uses modular arithmetic to avoid overflow: results mod 10**9 + 7.

    Args:
        n: Grid dimension (n x n).

    Returns:
        Tuple of (paths_to_bottom_right mod M, paths_to_center mod M).
    """
    MOD = 1000000007

    # Build obstacle grid
    blocked = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if (i * 7 + j * 13 + 3) % 17 == 0:
                blocked[i][j] = True
    # Ensure start and end are clear
    blocked[0][0] = False
    blocked[n - 1][n - 1] = False

    dp = [[0] * n for _ in range(n)]
    dp[0][0] = 1

    # First row
    for j in range(1, n):
        if not blocked[j][0]:
            dp[j][0] = dp[j - 1][0]

    # First column
    for j in range(1, n):
        if not blocked[0][j]:
            dp[0][j] = dp[0][j - 1]

    # Fill table
    for i in range(1, n):
        for j in range(1, n):
            if not blocked[i][j]:
                dp[i][j] = (dp[i - 1][j] + dp[i][j - 1]) % MOD

    mid = n // 2
    return (dp[n - 1][n - 1], dp[mid][mid])
