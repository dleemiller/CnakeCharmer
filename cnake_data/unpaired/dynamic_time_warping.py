"""Standard DTW distance and path extraction."""

from __future__ import annotations

import numpy as np


def dtw_std(x, y, dist_only=True, squared=False):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n, m = len(x), len(y)

    cost = np.full((n + 1, m + 1), np.inf)
    cost[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d = x[i - 1] - y[j - 1]
            local = d * d if squared else abs(d)
            cost[i, j] = local + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])

    dist = cost[n, m]
    if dist_only:
        return dist

    i, j = n, m
    path_i, path_j = [n - 1], [m - 1]
    while i > 1 or j > 1:
        prev = np.argmin([cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1]])
        if prev == 0:
            i -= 1
        elif prev == 1:
            j -= 1
        else:
            i -= 1
            j -= 1
        path_i.append(i - 1)
        path_j.append(j - 1)

    path_i.reverse()
    path_j.reverse()
    return dist, cost[1:, 1:], (np.array(path_i), np.array(path_j))
