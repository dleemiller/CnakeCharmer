from __future__ import annotations


def global_alignment_score(
    a: str, b: str, match: int = 2, mismatch: int = -1, gap: int = -2
) -> int:
    n = len(a)
    m = len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = dp[i - 1][0] + gap
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j - 1] + gap

    for i in range(1, n + 1):
        ai = a[i - 1]
        for j in range(1, m + 1):
            s = match if ai == b[j - 1] else mismatch
            d = dp[i - 1][j - 1] + s
            u = dp[i - 1][j] + gap
            l = dp[i][j - 1] + gap
            dp[i][j] = max(d, u, l)
    return dp[n][m]
