"""1D reflected-path importance-sampling simulation helpers."""

from __future__ import annotations

import math
import random


def simulation_diff(x_in, x_r, x_end, t_i, t_f, dt, bias, drift, diffusion, count_refs=False):
    t = t_i
    x = x_in
    a = 0.0
    refs = 0
    while t < t_f and x < x_end:
        d_w = random.gauss(0.0, math.sqrt(dt))
        x_new = x + drift(x, t) * dt + diffusion(x, t) * d_w + bias(x, t) * dt
        if x_new <= x_r:
            x_new = 2 * x_r - x_new
            refs += 1
        b = bias(x, t)
        a += b * d_w - 0.5 * b * b * dt
        x = x_new
        t += dt
    w = math.exp(-a)
    return (t, w, refs) if count_refs else (t, w)


def importance_sampling_simulations(
    x_in, x_r, x_end, t_i, t_f, dt, bias, num_runs, drift, diffusion, count_refs=False
):
    results = [
        simulation_diff(
            x_in, x_r, x_end, t_i, t_f, dt, bias, drift, diffusion, count_refs=count_refs
        )
        for _ in range(num_runs)
    ]
    if not count_refs:
        ts = [r[0] for r in results]
        ws = [r[1] for r in results]
        return ts, ws
    ts = [r[0] for r in results]
    ws = [r[1] for r in results]
    rs = [r[2] for r in results]
    return ts, ws, rs
