"""
Length of longest palindromic subsequence using DP.

Keywords: dynamic programming, palindrome, subsequence, string, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(3000,))
def longest_palindrome(n: int) -> int:
    """Find the length of the longest palindromic subsequence.

    String: s[i] = chr(65 + (i*7+3) % 26) for i in 0..n-1.
    Uses two-row DP to compute LPS length.

    Args:
        n: Length of the string.

    Returns:
        Length of the longest palindromic subsequence.
    """
    # Generate string as integer codes
    s = [(i * 7 + 3) % 26 for i in range(n)]

    # dp[j] = LPS length for substring s[i..j]
    # Two rows: prev = dp[i+1], curr = dp[i]
    # Iterate i from n-1 down to 0, j from i+1 to n-1
    prev = [0] * n
    curr = [0] * n

    for i in range(n - 1, -1, -1):
        curr[i] = 1  # base case: single char
        for j in range(i + 1, n):
            if s[i] == s[j]:
                curr[j] = prev[j - 1] + 2
            else:
                a = prev[j]  # dp[i+1][j]
                b = curr[j - 1]  # dp[i][j-1]
                curr[j] = a if a > b else b
        # Swap: prev becomes current row for next iteration
        prev, curr = curr, prev

    # After last swap, result is in prev (which was the last curr)
    return prev[n - 1]
