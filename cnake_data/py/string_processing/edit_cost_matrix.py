"""
Edit distance cost matrix.

Computes the full DP matrix for transforming string s into string t,
with configurable substitution cost.

Keywords: string processing, edit distance, cost matrix, dynamic programming, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(500, 2))
def edit_cost_matrix(n: int, substitution_cost: int) -> tuple:
    """Compute the full edit-distance cost matrix for two deterministic strings.

    Args:
        n: Controls string length. s has length n, t has length n+n//5.
        substitution_cost: Cost of a substitution operation.

    Returns:
        (edit_distance, cost_at_mid, checksum_of_last_row)
    """
    # Generate deterministic strings using modular arithmetic
    s = bytes([(i * 7 + 3) % 26 + 97 for i in range(n)])
    t = bytes([(i * 11 + 5) % 26 + 97 for i in range(n + n // 5)])
    m = len(s)
    p = len(t)

    # Build full (m+1) x (p+1) cost matrix as nested lists
    dp = [[0] * (p + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(p + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, p + 1):
            if s[i - 1] == t[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(
                    dp[i - 1][j] + 1,  # deletion
                    dp[i][j - 1] + 1,  # insertion
                    dp[i - 1][j - 1] + substitution_cost,  # substitution
                )

    edit_dist = dp[m][p]
    cost_at_mid = dp[m // 2][p // 2]
    checksum = 0
    for j in range(p + 1):
        checksum += dp[m][j]

    return (edit_dist, cost_at_mid, checksum)
