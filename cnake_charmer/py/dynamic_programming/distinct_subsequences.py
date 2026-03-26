"""
Count distinct subsequences of one string in another using DP.

Keywords: dynamic programming, distinct subsequences, string, counting, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(1500,))
def distinct_subsequences(n: int) -> tuple:
    """Count distinct subsequences of t in s using DP.

    s has length n, t has length n//5.
    s[i] = chr(97 + (i*7+3) % 3), t[j] = s[j*5] (every 5th char of s).
    Using alphabet of 3 chars ensures many matches. t is a subsequence of s
    by construction, so count >= 1.
    Results computed modulo 10**9 + 7.

    Args:
        n: Length of string s (t is n//5).

    Returns:
        Tuple of (count mod M, dp_mid_value mod M).
    """
    MOD = 1000000007
    len_s = n
    len_t = n // 5

    s = [chr(97 + (i * 7 + 3) % 3) for i in range(len_s)]
    t = [s[j * 5] for j in range(len_t)]

    # dp[i][j] = number of distinct subsequences of t[0..j-1] in s[0..i-1]
    # Use 1D rolling array: dp[j] for current row
    dp_prev = [0] * (len_t + 1)
    dp_curr = [0] * (len_t + 1)
    dp_prev[0] = 1
    dp_curr[0] = 1

    # Store a mid-table value for fingerprinting
    mid_s = len_s // 2
    mid_t = len_t // 2
    mid_val = 0

    for i in range(1, len_s + 1):
        dp_curr[0] = 1
        for j in range(1, len_t + 1):
            dp_curr[j] = dp_prev[j] % MOD
            if s[i - 1] == t[j - 1]:
                dp_curr[j] = (dp_curr[j] + dp_prev[j - 1]) % MOD
        if i == mid_s:
            mid_val = dp_curr[mid_t]
        # Swap rows
        dp_prev, dp_curr = dp_curr, dp_prev

    return (dp_prev[len_t], mid_val)
