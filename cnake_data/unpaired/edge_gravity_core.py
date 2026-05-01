from __future__ import annotations

from collections import defaultdict


def is_subsequence_edge(edge: tuple[str, str], path: tuple[str, ...]) -> bool:
    n = len(path)
    return any(path[i] == edge[0] and path[i + 1] == edge[1] for i in range(n - 1))


def accumulate_edge_hits(
    edges: list[tuple[str, str]], shortest_paths: list[tuple[str, str, tuple[str, ...]]]
) -> dict[tuple[str, str], int]:
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for _, _, p in shortest_paths:
        for e in edges:
            if is_subsequence_edge(e, p):
                counts[e] += 1
    return counts
