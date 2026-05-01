"""Dijkstra shortest path on weighted adjacency matrix."""

from __future__ import annotations


def get_root(start, stop, previous):
    v = stop
    root = []
    while True:
        root.append(v)
        if v == start:
            break
        v = previous[v]
    return root


def dijkstra(w, start, stop, n):
    p = [float("inf")] * n
    t = [float("inf")] * n
    pr = [0] * n
    s = [0.0] * n
    p[start] = 0.0
    v = start

    while v != stop:
        for j in range(n):
            s[j] = p[v] + w[v][j]
        for j in range(n):
            if s[j] < t[j]:
                pr[j] = v
        for j in range(n):
            t[j] = min(t[j], s[j])
        i = t.index(min(t))
        p[i] = t[i]
        t[i] = float("inf")
        for j in range(n):
            w[j][i] = float("inf")
        v = i

    return p[v], get_root(start, stop, pr)
