"""Approximate muon-decay positron spectrum kernels."""

from __future__ import annotations

import math

import numpy as np

MASS_E = 0.510998950e-3
MASS_MU = 0.1056583755
R = MASS_E / MASS_MU
R2 = R * R
R_FACTOR = 1.0001870858234163


def dndx_positron_muon_rest_frame(x):
    if x <= 2 * R or x >= 1.0 + R2:
        return 0.0
    return (
        -2.0 * math.sqrt(x**2 - 4.0 * R2) * (4.0 * R2 + x * (-3.0 - 3.0 * R2 + 2.0 * x)) / R_FACTOR
    )


def dndx_positron_muon(x, beta):
    if beta < 0.0 or beta > 1.0:
        return 0.0
    if beta < np.finfo(float).eps:
        return dndx_positron_muon_rest_frame(x)

    gamma2 = 1.0 / (1.0 - beta**2)
    r22 = 4.0 * R2 * (1.0 - beta**2)

    xm = max(gamma2 * (x - beta * math.sqrt(max(x**2 - r22, 0.0))), 2.0 * R)
    xp = min(gamma2 * (x + beta * math.sqrt(max(x**2 - r22, 0.0))), 1.0 + R2)
    if xm > xp:
        return 0.0

    return (
        xm * (8 * R2 + xm * (-3 - 3 * R2 + (4 * xm) / 3.0))
        + xp * (-8 * R2 + (3 + 3 * R2 - (4 * xp) / 3.0) * xp)
    ) / (2 * beta * R_FACTOR)


def dnde_positron_muon_point(e, emu):
    if emu < MASS_MU or e <= MASS_E:
        return 0.0

    if emu - MASS_MU < np.finfo(float).eps:
        pre = 2.0 / MASS_MU
        dndx = dndx_positron_muon_rest_frame(pre * e)
    else:
        beta = math.sqrt(max(1.0 - (MASS_MU / emu) ** 2, 0.0))
        pre = 2.0 / emu
        dndx = dndx_positron_muon(pre * e, beta)

    return pre * dndx


def dnde_positron_muon_array(engs_p, eng_mu):
    out = np.empty(len(engs_p), dtype=float)
    for i, e in enumerate(engs_p):
        out[i] = dnde_positron_muon_point(e, eng_mu)
    return out
