import math


def from_distance(D):
    """Compute DTW dynamic table from distance matrix D."""
    m = len(D)
    n = len(D[0]) if m else 0
    dtw = [[math.inf] * (n + 1) for _ in range(m + 1)]
    dtw[0][0] = 0.0

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dtw[i][j] = D[i - 1][j - 1] + min(dtw[i - 1][j], dtw[i][j - 1], dtw[i - 1][j - 1])
    return dtw


def path(dtw):
    """Backtrack a DTW table to align sequence indices."""
    m = len(dtw) - 1
    n = len(dtw[0]) - 1
    i = m - 1
    j = n - 1
    s = [0] * m
    t = [0] * n

    while i >= 0 or j >= 0:
        if i >= 0:
            s[i] = max(j, 0)
        if j >= 0:
            t[j] = max(i, 0)

        up = dtw[i][j + 1] if i >= 0 and j + 1 < len(dtw[0]) else math.inf
        left = dtw[i + 1][j] if j >= 0 and i + 1 < len(dtw) else math.inf
        diag = dtw[i][j] if i >= 0 and j >= 0 else math.inf

        if up < left:
            if up < diag:
                i -= 1
            else:
                i -= 1
                j -= 1
        elif left < diag:
            j -= 1
        else:
            i -= 1
            j -= 1

    return s, t


def match(s, D):
    """Resolve many-to-one alignments by keeping minimal local distance."""
    s = s[:]
    n = len(s)

    i = 0
    while i < n:
        j = s[i]
        k = i
        m = i
        best = D[i][j] if j >= 0 else math.inf
        while k < n and s[k] == j:
            if D[k][j] < best:
                m = k
                best = D[k][j]
            k += 1
        kk = i
        while kk < n and s[kk] == j:
            if kk != m:
                s[kk] = -1
            kk += 1
        i = k

    return s
