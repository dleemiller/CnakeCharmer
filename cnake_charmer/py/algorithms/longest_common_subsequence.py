"""Longest common subsequence length of two deterministic sequences.

Keywords: algorithms, LCS, longest common subsequence, dynamic programming, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(2000,))
def longest_common_subsequence(n: int) -> int:
    """Compute LCS length of two sequences.

    Sequences: a[i] = (i*7+3) % 20, b[i] = (i*11+5) % 20.
    Uses standard DP approach with flat table.

    Args:
        n: Length of each sequence.

    Returns:
        Length of the longest common subsequence.
    """
    a = [(i * 7 + 3) % 20 for i in range(n)]
    b = [(i * 11 + 5) % 20 for i in range(n)]

    # Flat DP table: dp[i*(n+1)+j] = LCS length of a[:i] and b[:j]
    dp = [0] * ((n + 1) * (n + 1))

    for i in range(1, n + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i * (n + 1) + j] = dp[(i - 1) * (n + 1) + (j - 1)] + 1
            else:
                val1 = dp[(i - 1) * (n + 1) + j]
                val2 = dp[i * (n + 1) + (j - 1)]
                dp[i * (n + 1) + j] = val1 if val1 > val2 else val2

    return dp[n * (n + 1) + n]
