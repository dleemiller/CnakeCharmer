from __future__ import annotations


def choose_min_queue(lengths: list[int]) -> int:
    if not lengths:
        raise ValueError("empty lengths")
    best_i = 0
    best_v = lengths[0]
    for i in range(1, len(lengths)):
        if lengths[i] < best_v:
            best_v = lengths[i]
            best_i = i
    return best_i


def choose_weighted(lengths: list[int], rng01: float, eps: float = 1e-9) -> int:
    """Choose index with probability proportional to 1/(len+eps)."""
    w = [1.0 / (v + eps) for v in lengths]
    z = sum(w)
    t = rng01 * z
    s = 0.0
    for i, wi in enumerate(w):
        s += wi
        if t <= s:
            return i
    return len(lengths) - 1
