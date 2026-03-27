"""Count valid parenthesizations of length 2n (Catalan numbers via DP).

Keywords: leetcode, parentheses, catalan, dynamic programming, combinatorics, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(800,))
def valid_parentheses_gen(n: int) -> tuple:
    """Count the number of valid parenthesizations of length 2n.

    Uses a 2D DP table where dp[i][j] = number of valid sequences of length i
    with j unmatched open parens. The answer is dp[2n][0] which equals the
    nth Catalan number. Uses modular arithmetic with mod = 10**9 + 7.

    Args:
        n: Half-length of parenthesization (total length = 2n).

    Returns:
        Tuple of (count_mod, dp_mid_value, dp_quarter_value).
    """
    mod = 1000000007
    total = 2 * n

    # dp[i][j] = ways to form valid prefix of length i with j open parens
    # Flat array: dp[i * (n+1) + j]
    size = (total + 1) * (n + 1)
    dp = [0] * size

    dp[0 * (n + 1) + 0] = 1  # empty string, 0 open parens

    for i in range(total):
        for j in range(n + 1):
            val = dp[i * (n + 1) + j]
            if val == 0:
                continue
            # Add '(' -> j+1 open parens
            if j + 1 <= n:
                dp[(i + 1) * (n + 1) + j + 1] = (dp[(i + 1) * (n + 1) + j + 1] + val) % mod
            # Add ')' -> j-1 open parens
            if j > 0:
                dp[(i + 1) * (n + 1) + j - 1] = (dp[(i + 1) * (n + 1) + j - 1] + val) % mod

    count = dp[total * (n + 1) + 0]

    # Diagnostic: sum of dp values along the diagonal dp[i][i//2]
    dp_diag_sum = 0
    for i in range(1, total + 1):
        j = i // 2
        if j <= n:
            dp_diag_sum = (dp_diag_sum + dp[i * (n + 1) + j]) % mod

    # Diagnostic: dp value at (n, n%2) -- halfway through with matching parity
    # n and n%2 have same parity, so (n + n%2)/2 is integer -> guaranteed non-zero for n>=2
    parity_j = n % 2
    mid_val = dp[n * (n + 1) + parity_j]

    return (count, dp_diag_sum, mid_val)
