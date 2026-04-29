"""Muon-decay neutrino spectrum in rest and boosted frames."""

from __future__ import annotations

import math

import numpy as np

MASS_E = 0.51099895
MASS_MU = 105.6583755
R = MASS_E / MASS_MU
R2 = R * R
R4 = R2 * R2
R6 = R4 * R2
R_FACTOR = 1.0001870858234163


def muon_decay_spectrum_point_rest(enu: float) -> tuple[float, float, float]:
    pre = 2.0 / MASS_MU
    x = pre * enu
    if x <= 0.0 or x >= 1 - R**2:
        return 0.0, 0.0, 0.0
    xm = 1.0 - x
    common = R_FACTOR * x**2 * (1.0 - R**2 - x) ** 2 / xm
    dndxe = 12.0 * common
    dndxm = 2.0 * common * (3.0 + R2 * (3.0 - x) - 5.0 * x + 2.0 * x**2) / xm**2
    return pre * dndxe, pre * dndxm, 0.0


def muon_decay_spectrum_point(enu: float, emu: float) -> tuple[float, float, float]:
    if emu < MASS_MU:
        return 0.0, 0.0, 0.0
    if emu - MASS_MU < 1e-15:
        return muon_decay_spectrum_point_rest(enu)

    e_to_x = 2.0 / emu
    x = e_to_x * enu
    gam = emu / MASS_MU
    beta = math.sqrt(1.0 - (MASS_MU / emu) ** 2)
    pre = R_FACTOR * e_to_x / (2.0 * beta)
    xmax_rf = 1 - R**2

    if x <= 0.0 or (1.0 + beta) * xmax_rf <= x:
        return 0.0, 0.0, 0.0

    xm = gam**2 * x * (1.0 - beta)
    xp = min(xmax_rf, gam**2 * x * (1.0 + beta))
    xmm = 1.0 - xm
    xpm = 1.0 - xp

    electron = (
        2
        * pre
        * (
            (xm - xp)
            * (-3.0 * (xm + xp) + 2 * (3 * R4 + xm**2 + xm * xp + xp**2 + 3 * R2 * (xm + xp)))
            - 6 * R4 * math.log(xpm / xmm)
        )
    )
    muon = pre * (
        3 * R2 * (xm - xp) * (xm + xp)
        + (xm**2 * (-9.0 + 4.0 * xm) + (9.0 - 4 * xp) * xp**2) / 3.0
        + R6 * ((-2.0 * xm) / xmm**2 + (2.0 * xp) / xpm**2)
        + 6 * R4 * (1.0 / xmm - 1.0 / xpm)
        + 2 * R4 * (-3 + R2) * math.log(xpm / xmm)
    )
    return electron, muon, 0.0


def dnde_neutrino_muon(egam, emu: float):
    if hasattr(egam, "__len__"):
        energies = np.asarray(egam, dtype=float)
        spec = np.zeros((3, len(energies)), dtype=float)
        for i, e in enumerate(energies):
            spec[0, i], spec[1, i], spec[2, i] = muon_decay_spectrum_point(float(e), emu)
        return spec
    return muon_decay_spectrum_point(float(egam), emu)
