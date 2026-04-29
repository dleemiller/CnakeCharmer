"""Floyd-Warshall shortest path with footprint reconstruction."""

from __future__ import annotations


def dynamic_programming_to_find_the_shortest_path(
    from_vertex, to_vertex, path, n, dist, foot_print
):
    for k in range(n):
        for i in range(n):
            dik = dist[i][k]
            for j in range(n):
                if dist[i][j] >= dik + dist[k][j]:
                    dist[i][j] = dik + dist[k][j]
                    if dist[i][j] < float("inf"):
                        foot_print[i][j] = foot_print[i][k]

    if foot_print[from_vertex][to_vertex] == -1:
        return []

    idx = 0
    path[idx] = from_vertex
    idx += 1
    while from_vertex != to_vertex:
        from_vertex = foot_print[from_vertex][to_vertex]
        path[idx] = from_vertex
        idx += 1
    return path
