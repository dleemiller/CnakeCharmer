from __future__ import annotations


def edge_cut_score(
    partition: list[int], edges: list[tuple[int, int]], weights: list[float]
) -> float:
    if len(edges) != len(weights):
        raise ValueError("edges/weights mismatch")
    s = 0.0
    for (u, v), w in zip(edges, weights, strict=False):
        if partition[u] != partition[v]:
            s += w
    return s


def balance_penalty(partition: list[int], n_parts: int) -> float:
    counts = [0] * n_parts
    for p in partition:
        counts[p] += 1
    mean = len(partition) / max(1, n_parts)
    pen = 0.0
    for c in counts:
        d = c - mean
        pen += d * d
    return pen / max(1, len(partition))


def objective(
    partition: list[int],
    edges: list[tuple[int, int]],
    weights: list[float],
    n_parts: int,
    lam: float,
) -> float:
    return edge_cut_score(partition, edges, weights) - lam * balance_penalty(partition, n_parts)
