"""Count ways to break a generated string into words of length 1-5.

Keywords: word break, dynamic programming, string, counting, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(10000,))
def word_break(n: int) -> int:
    """Count ways to segment a string of n characters into words of length 1-5.

    String is generated as: s[i] = chr(65 + (i*7+3) % 5).
    A word can be any substring of length 1 to 5.

    Args:
        n: Length of the string.

    Returns:
        Number of ways to break the string, mod 10^9+7.
    """
    MOD = 1000000007

    # dp[i] = number of ways to segment s[0:i]
    dp = [0] * (n + 1)
    dp[0] = 1  # empty string: one way

    for i in range(1, n + 1):
        for length in range(1, 6):  # word lengths 1 to 5
            if i - length >= 0:
                dp[i] = (dp[i] + dp[i - length]) % MOD

    return dp[n]
