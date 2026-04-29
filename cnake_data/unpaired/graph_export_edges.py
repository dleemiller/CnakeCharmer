from __future__ import annotations


def edge_list_from_adjacency(adj: list[list[int]]) -> list[tuple[int, int]]:
    edges: list[tuple[int, int]] = []
    n = len(adj)
    for i in range(n):
        row = adj[i]
        for j in range(i + 1, len(row)):
            if row[j]:
                edges.append((i, j))
    return edges


def degree_counts(adj: list[list[int]]) -> list[int]:
    return [sum(1 for v in row if v) for row in adj]
