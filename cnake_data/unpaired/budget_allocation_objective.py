"""Budget-allocation graph objective kernels."""

from __future__ import annotations

import numpy as np


def construct_g_p(graph):
    max_degree = max(len(graph[v]) for v in graph)
    g_array = np.full((len(graph), max_degree), -1, dtype=int)
    p = np.zeros((len(graph), max_degree), dtype=float)
    for v in range(len(graph)):
        for i, (u, prob) in enumerate(graph[v].items()):
            g_array[v, i] = u
            p[v, i] = prob
    return g_array, p


def budget_objective(g, p, weights, y):
    total = 0.0
    for v in range(g.shape[0]):
        p_fail = 1.0
        for i in range(g.shape[1]):
            u = g[v, i]
            if u == -1:
                break
            p_fail *= (1.0 - p[v, i]) ** y[u]
        total += weights[v] * (1.0 - p_fail)
    return total
