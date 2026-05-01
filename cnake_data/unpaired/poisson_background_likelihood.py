"""Poisson log-likelihood with phase-shifted component integration."""

from __future__ import annotations

import math


def _integrate_piecewise_linear(x: list[float], y: list[float], a: float, b: float) -> float:
    if b <= a:
        return 0.0
    acc = 0.0
    for i in range(len(x) - 1):
        left = max(a, x[i])
        right = min(b, x[i + 1])
        if right <= left:
            continue
        t0 = (left - x[i]) / (x[i + 1] - x[i])
        t1 = (right - x[i]) / (x[i + 1] - x[i])
        yl = y[i] + t0 * (y[i + 1] - y[i])
        yr = y[i] + t1 * (y[i + 1] - y[i])
        acc += 0.5 * (yl + yr) * (right - left)
    return acc


def poisson_likelihood_given_background(
    exposure_time: float,
    phases: list[float],
    counts: list[list[float]],
    components: list[list[list[float]]],
    component_phases: list[list[float]],
    phase_shifts: list[float],
    background: list[list[float]],
    allow_negative: bool | list[bool] = False,
) -> tuple[float, list[list[float]]]:
    n_channels = len(components[0])
    n_bins = len(phases) - 1
    n_components = len(components)

    star = [[0.0 for _ in range(n_bins)] for _ in range(n_channels)]

    if isinstance(allow_negative, bool):
        allow = [allow_negative] * n_components
    else:
        allow = list(allow_negative)

    for i in range(n_channels):
        for p in range(n_components):
            signal = components[p][i]
            signal_phase_set = component_phases[p]
            phase_shift = phase_shifts[p]

            for j in range(n_bins):
                a = phases[j] + phase_shift
                b = phases[j + 1] + phase_shift

                if b - a == 1.0:
                    a, b = 0.0, 1.0
                else:
                    a = a - math.floor(a)
                    b = b - math.floor(b)

                if a < b:
                    val = _integrate_piecewise_linear(signal_phase_set, signal, a, b)
                    if val > 0.0 or allow[p]:
                        star[i][j] += val
                else:
                    v1 = _integrate_piecewise_linear(signal_phase_set, signal, a, 1.0)
                    v2 = _integrate_piecewise_linear(signal_phase_set, signal, 0.0, b)
                    if v1 > 0.0 or allow[p]:
                        star[i][j] += v1
                    if v2 > 0.0 or allow[p]:
                        star[i][j] += v2

    log_like = 0.0
    for i in range(n_channels):
        for j in range(n_bins):
            lam = max(1e-300, exposure_time * star[i][j] + background[i][j])
            k = counts[i][j]
            log_like += k * math.log(lam) - lam - math.lgamma(k + 1.0)

    return log_like, star
