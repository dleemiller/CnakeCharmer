from __future__ import annotations

import heapq


def dijkstra(
    n_vertices: int, edges: list[tuple[int, int, float]], source: int
) -> tuple[list[int | None], list[float | None]]:
    g: list[list[tuple[int, float]]] = [[] for _ in range(n_vertices)]
    for u, v, w in edges:
        g[u].append((v, w))
    dist = [float("inf")] * n_vertices
    prev: list[int | None] = [None] * n_vertices
    dist[source] = 0.0
    pq: list[tuple[float, int]] = [(0.0, source)]

    while pq:
        d, u = heapq.heappop(pq)
        if d != dist[u]:
            continue
        for v, w in g[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))

    dist_out: list[float | None] = [None if x == float("inf") else x for x in dist]
    return prev, dist_out
