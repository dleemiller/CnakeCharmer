"""
2D prefix sums (summed area table).

Keywords: numerical, prefix sum, 2D, summed area table, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(1000,))
def prefix_sum_2d(n: int) -> tuple:
    """Compute 2D prefix sums for an n x n grid.

    Grid: grid[i][j] = ((i * 1009 + j * 2003 + 42) * 17 + 137) % 256
    Prefix sum: P[i][j] = sum of grid[r][c] for r in 0..i, c in 0..j

    Args:
        n: Grid dimension (n x n).

    Returns:
        (P[n-1][n-1], P[n//2][n//3], P[n//3][n//2], P[n-1][0] + P[0][n-1])
    """
    # Build grid and prefix sum table as 2D lists
    P = [[0] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            g = ((i * 1009 + j * 2003 + 42) * 17 + 137) % 256
            above = P[i - 1][j] if i > 0 else 0
            left = P[i][j - 1] if j > 0 else 0
            diag = P[i - 1][j - 1] if (i > 0 and j > 0) else 0
            P[i][j] = g + above + left - diag

    return (
        P[n - 1][n - 1],
        P[n // 2][n // 3],
        P[n // 3][n // 2],
        P[n - 1][0] + P[0][n - 1],
    )
