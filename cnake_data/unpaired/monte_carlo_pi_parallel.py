"""Parallel Monte Carlo Pi estimation."""

from __future__ import annotations

import random
from concurrent.futures import ProcessPoolExecutor


def mc_pi_once(n: int, seed: int | None = None) -> float:
    rng = random.Random(seed)
    inside = 0
    for _ in range(n):
        x = 2.0 * rng.random() - 1.0
        y = 2.0 * rng.random() - 1.0
        if x * x + y * y < 1.0:
            inside += 1
    return 4.0 * inside / n


def mc_pi_parallel(n_per_job: int = 10000, jobs: int = 32, max_workers: int = 8) -> float:
    seeds = list(range(jobs))
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        vals = list(pool.map(mc_pi_once, [n_per_job] * jobs, seeds))
    return sum(vals) / len(vals)
