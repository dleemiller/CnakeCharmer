from __future__ import annotations


def advance_state(
    pos: list[float],
    vel: list[float],
    acc: list[float],
    dt: float,
    steps: int,
) -> tuple[list[float], list[float]]:
    """Simple explicit time integration for independent 1D state vectors."""
    if not (len(pos) == len(vel) == len(acc)):
        raise ValueError("pos, vel, acc must have identical lengths")

    p = pos[:]
    v = vel[:]
    for _ in range(steps):
        for i in range(len(p)):
            v[i] += acc[i] * dt
            p[i] += v[i] * dt
    return p, v


def timestep_error(reference: list[float], estimate: list[float]) -> float:
    if len(reference) != len(estimate):
        raise ValueError("length mismatch")
    s = 0.0
    for i in range(len(reference)):
        d = reference[i] - estimate[i]
        s += d * d
    return s / max(1, len(reference))
