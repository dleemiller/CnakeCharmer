"""Compute edit distance with full alignment traceback.

Keywords: dynamic programming, edit distance, levenshtein, alignment, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(1500,))
def edit_distance_full(n: int) -> tuple:
    """Compute edit distance between two deterministic strings of length n.

    String a: a[i] = 'a' + (i * 7 + 3) % 26
    String b: b[i] = 'a' + (i * 13 + 5) % 26
    Uses O(n^2) DP, then traces back to count substitutions in optimal alignment.

    Args:
        n: Length of each string.

    Returns:
        Tuple of (distance, dp_mid_mid, num_substitutions_in_alignment).
    """
    # Generate strings as integer arrays (char codes)
    a = [0] * n
    b = [0] * n
    for i in range(n):
        a[i] = (i * 7 + 3) % 26
        b[i] = (i * 13 + 5) % 26

    # DP table: dp[(i)*(n+1) + j] = edit distance of a[:i] and b[:j]
    sz = (n + 1) * (n + 1)
    dp = [0] * sz
    stride = n + 1

    for i in range(n + 1):
        dp[i * stride + 0] = i
    for j in range(n + 1):
        dp[0 * stride + j] = j

    for i in range(1, n + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i * stride + j] = dp[(i - 1) * stride + (j - 1)]
            else:
                sub = dp[(i - 1) * stride + (j - 1)] + 1
                ins = dp[i * stride + (j - 1)] + 1
                dlt = dp[(i - 1) * stride + j] + 1
                best = sub
                if ins < best:
                    best = ins
                if dlt < best:
                    best = dlt
                dp[i * stride + j] = best

    distance = dp[n * stride + n]

    mid = n // 2
    dp_mid_mid = dp[mid * stride + mid]

    # Traceback to count substitutions
    num_subs = 0
    i = n
    j = n
    while i > 0 and j > 0:
        dp[i * stride + j]
        diag = dp[(i - 1) * stride + (j - 1)]
        up = dp[(i - 1) * stride + j]
        left = dp[i * stride + (j - 1)]

        if a[i - 1] == b[j - 1]:
            i -= 1
            j -= 1
        elif diag <= up and diag <= left:
            num_subs += 1
            i -= 1
            j -= 1
        elif up <= left:
            i -= 1
        else:
            j -= 1

    return (distance, dp_mid_mid, num_subs)
