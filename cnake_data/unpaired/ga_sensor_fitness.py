from __future__ import annotations


def coverage_score(
    sensors: list[tuple[float, float]],
    targets: list[tuple[float, float]],
    radius: float,
) -> int:
    r2 = radius * radius
    covered = 0
    for tx, ty in targets:
        hit = False
        for sx, sy in sensors:
            dx = tx - sx
            dy = ty - sy
            if dx * dx + dy * dy <= r2:
                hit = True
                break
        if hit:
            covered += 1
    return covered


def chromosome_fitness(
    chromosome: list[int],
    candidates: list[tuple[float, float]],
    targets: list[tuple[float, float]],
    radius: float,
    active_penalty: float,
) -> float:
    sensors = [candidates[i] for i, bit in enumerate(chromosome) if bit]
    cov = coverage_score(sensors, targets, radius)
    return cov - active_penalty * len(sensors)
