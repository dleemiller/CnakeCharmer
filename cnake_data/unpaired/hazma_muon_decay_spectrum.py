"""Muon radiative decay spectrum helpers (standalone approximation)."""

from __future__ import annotations

from collections.abc import Iterable
from math import log, pi, sqrt

import numpy as np

ALPHA_EM = 1.0 / 137.035999084
MASS_MU = 0.1056583745
MASS_E = 0.00051099895


def muon_decay_spectrum_point_rest(egam: float) -> float:
    y = 2.0 * egam / MASS_MU
    r = (MASS_E / MASS_MU) ** 2
    if y <= 0.0 or y >= 1.0 - MASS_E / MASS_MU:
        return 0.0
    pre = ALPHA_EM / (3.0 * pi * y * MASS_MU)
    ym = 1.0 - y
    poly1 = -102.0 + 46.0 * y - 101.0 * y * y + 55.0 * y**3
    poly2 = 3.0 - 5.0 * y + 6.0 * y * y - 6.0 * y**3 + 2.0 * y**4
    return pre * (poly1 * ym / 12.0 + poly2 * log(ym / r))


def _simpson_integral(fn, a: float, b: float, n: int = 256) -> float:
    if n % 2 == 1:
        n += 1
    h = (b - a) / n
    s = fn(a) + fn(b)
    for i in range(1, n):
        x = a + i * h
        s += (4.0 if i % 2 else 2.0) * fn(x)
    return s * h / 3.0


def muon_decay_spectrum_point(egam: float, emu: float) -> float:
    if emu < MASS_MU:
        return 0.0
    if abs(emu - MASS_MU) < 1e-12:
        return muon_decay_spectrum_point_rest(egam)

    gamma = emu / MASS_MU
    beta = sqrt(max(0.0, 1.0 - (MASS_MU / emu) ** 2))
    y = 2.0 * egam / MASS_MU
    r = (MASS_E / MASS_MU) ** 2
    x = y * gamma

    upper = (1.0 - r) / (1.0 - beta)
    if x < 0.0 or x >= upper:
        return 0.0

    def integrand(w: float) -> float:
        xr = x * w
        if xr <= 0.0 or xr >= 1.0 - r:
            return 0.0
        val = muon_decay_spectrum_point_rest(0.5 * xr * MASS_MU)
        return val / (2.0 * gamma * w)

    w_min = 1.0 - beta
    w_max = min(1.0 + beta, (1.0 - r) / x)
    return _simpson_integral(integrand, w_min, w_max)


def muon_decay_spectrum(egam: float | Iterable[float], emu: float) -> float | np.ndarray:
    if hasattr(egam, "__len__"):
        energies = np.asarray(list(egam), dtype=np.float64)
        if energies.ndim != 1:
            raise ValueError("Photon energies must be 0 or 1-dimensional.")
        return np.array(
            [muon_decay_spectrum_point(float(e), emu) for e in energies], dtype=np.float64
        )
    return muon_decay_spectrum_point(float(egam), emu)
