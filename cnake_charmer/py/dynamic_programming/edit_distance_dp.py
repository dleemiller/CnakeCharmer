"""
Edit distance between two deterministic strings using full DP table.

Keywords: dynamic programming, edit distance, levenshtein, dp table, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(1500,))
def edit_distance_dp(n: int) -> int:
    """Compute edit distance between two deterministic strings of length n.

    Uses a full n x n DP table (not space-optimized).
    s1[i] = chr(65 + (i*7+3) % 26), s2[i] = chr(65 + (i*11+5) % 26).

    Args:
        n: Length of each string.

    Returns:
        The edit distance as an integer.
    """
    s1 = [chr(65 + (i * 7 + 3) % 26) for i in range(n)]
    s2 = [chr(65 + (i * 11 + 5) % 26) for i in range(n)]

    # Full DP table (n+1) x (n+1)
    dp = [[0] * (n + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    return (dp[n][n], dp[n // 2][n // 2])
