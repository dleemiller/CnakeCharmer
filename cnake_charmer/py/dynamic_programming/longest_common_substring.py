"""
Longest common substring between two deterministic strings using DP.

Keywords: dynamic programming, longest common substring, string matching, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(1500,))
def longest_common_substring(n: int) -> tuple:
    """Find longest common substring length between two deterministic strings.

    s1[i] = ((i * 2654435761) >> 4) % 3, s2[i] = ((i * 1640531527) >> 4) % 3.
    Uses full (n+1) x (n+1) DP table. Small alphabet (3 chars) ensures
    frequent matches and non-trivial common substrings.

    Args:
        n: Length of each string.

    Returns:
        Tuple of (max_length, sum_of_dp_diagonal).
    """
    mask = 0xFFFFFFFF
    s1 = [(((i * 2654435761) & mask) >> 4) % 3 for i in range(n)]
    s2 = [(((i * 1640531527) & mask) >> 4) % 3 for i in range(n)]

    # Full DP table
    dp = [[0] * (n + 1) for _ in range(n + 1)]
    max_len = 0

    for i in range(1, n + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                if dp[i][j] > max_len:
                    max_len = dp[i][j]

    # Sum along main diagonal for discriminating fingerprint
    diag_sum = 0
    for i in range(1, n + 1):
        diag_sum += dp[i][i]

    return (max_len, diag_sum)
