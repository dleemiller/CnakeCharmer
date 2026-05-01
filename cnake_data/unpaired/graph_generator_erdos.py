from __future__ import annotations


def erdos_renyi_edges(n: int, probs: list[float], rnd: list[float]) -> list[tuple[int, int]]:
    """Generate undirected edges from pre-sampled uniforms.

    probs should contain one probability p (or per-edge probabilities).
    rnd provides uniforms for edges in lexicographic (i,j), i<j order.
    """
    if n < 0:
        raise ValueError("n must be nonnegative")
    edges: list[tuple[int, int]] = []
    k = 0
    default_p = probs[0] if probs else 0.0
    for i in range(n):
        for j in range(i + 1, n):
            p = probs[k] if k < len(probs) else default_p
            u = rnd[k] if k < len(rnd) else 1.0
            if u < p:
                edges.append((i, j))
            k += 1
    return edges
