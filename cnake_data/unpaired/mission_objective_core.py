from __future__ import annotations


def weighted_objective(values: list[float], weights: list[float], penalties: list[float]) -> float:
    if not (len(values) == len(weights) == len(penalties)):
        raise ValueError("length mismatch")
    total = 0.0
    for i in range(len(values)):
        v = values[i]
        w = weights[i]
        p = penalties[i]
        total += w * v - p * abs(v)
    return total


def best_action(
    actions: list[list[float]], weights: list[float], penalties: list[float]
) -> tuple[int, float]:
    best_i = -1
    best_s = float("-inf")
    for i, vec in enumerate(actions):
        s = weighted_objective(vec, weights, penalties)
        if s > best_s:
            best_s = s
            best_i = i
    return best_i, best_s
