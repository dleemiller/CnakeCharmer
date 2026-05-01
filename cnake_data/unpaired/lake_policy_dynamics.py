from __future__ import annotations


def anthropogenic_release(
    xt: float, c1: float, c2: float, r1: float, r2: float, w1: float
) -> float:
    v1 = abs((xt - c1) / r1)
    v2 = abs((xt - c2) / r2)
    rule = w1 * (v1**3) + (1.0 - w1) * (v2**3)
    if rule < 0.01:
        return 0.01
    if rule > 0.1:
        return 0.1
    return rule


def lake_step(x_prev: float, decision: float, inflow: float, b: float, q: float) -> float:
    return (1.0 - b) * x_prev + (x_prev**q) / (1.0 + x_prev**q) + decision + inflow


def simulate_lake(
    reps: int,
    steps: int,
    inflows: list[list[float]],
    b: float,
    q: float,
    c1: float,
    c2: float,
    r1: float,
    r2: float,
    w1: float,
) -> list[list[float]]:
    x = [[0.0 for _ in range(steps)] for _ in range(reps)]
    for r in range(reps):
        for t in range(1, steps):
            d = anthropogenic_release(x[r][t - 1], c1, c2, r1, r2, w1)
            x[r][t] = lake_step(x[r][t - 1], d, inflows[r][t - 1], b, q)
    return x
