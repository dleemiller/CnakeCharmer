from __future__ import annotations


def longest_common_substring_len(source: str, target: str) -> int:
    n = len(source)
    m = len(target)
    if n == 0 or m == 0:
        return 0
    table = [[0] * (m + 1) for _ in range(n + 1)]
    best = 0
    for i in range(1, n + 1):
        si = source[i - 1]
        for j in range(1, m + 1):
            if si == target[j - 1]:
                table[i][j] = table[i - 1][j - 1] + 1
                if table[i][j] > best:
                    best = table[i][j]
    return best


def longest_common_subsequence_len(source: str, target: str) -> int:
    n = len(source)
    m = len(target)
    table = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        si = source[i - 1]
        for j in range(1, m + 1):
            if si == target[j - 1]:
                table[i][j] = table[i - 1][j - 1] + 1
            else:
                table[i][j] = (
                    table[i - 1][j] if table[i - 1][j] >= table[i][j - 1] else table[i][j - 1]
                )
    return table[n][m]
