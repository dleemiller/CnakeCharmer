"""Long kaon radiative decay spectrum helpers.

Standalone Python adaptation of interpolation + boost integration logic.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from math import sqrt

import numpy as np

MASS_K0 = 0.497611


MODE_BITS = {
    "000": 1,
    "penu": 2,
    "penug": 4,
    "pm0": 8,
    "pm0g": 16,
    "pmunu": 32,
    "pmunug": 64,
}


def mode_bitflags(modes: Sequence[str]) -> int:
    flags = 0
    for name, bit in MODE_BITS.items():
        if name in modes:
            flags |= bit
    if flags == 0:
        raise ValueError("Invalid modes specified.")
    return flags


def interp_spec(
    eng_gam: float, bitflags: int, mode_tables: dict[str, tuple[np.ndarray, np.ndarray]]
) -> float:
    total = 0.0
    for name, bit in MODE_BITS.items():
        if bitflags & bit:
            energies, values = mode_tables[name]
            total += float(np.interp(eng_gam, energies, values))
    return total


def integrand(
    cl: float,
    eng_gam: float,
    eng_k: float,
    bitflags: int,
    mode_tables: dict[str, tuple[np.ndarray, np.ndarray]],
) -> float:
    gamma_k = eng_k / MASS_K0
    beta_k = sqrt(1.0 - (MASS_K0 / eng_k) ** 2)
    eng_rf = eng_gam * gamma_k * (1.0 - beta_k * cl)
    pre = 1.0 / (2.0 * gamma_k * (1.0 - beta_k * cl))
    return pre * interp_spec(eng_rf, bitflags, mode_tables)


def long_kaon_decay_spectrum_point(
    eng_gam: float,
    eng_k: float,
    bitflags: int,
    mode_tables: dict[str, tuple[np.ndarray, np.ndarray]],
    n_cl_samples: int = 256,
) -> float:
    if eng_k < MASS_K0:
        return 0.0
    grid = np.linspace(-1.0, 1.0, n_cl_samples)
    vals = np.array(
        [integrand(cl, eng_gam, eng_k, bitflags, mode_tables) for cl in grid], dtype=np.float64
    )
    return float(np.trapz(vals, grid))


def long_kaon_decay_spectrum(
    egam: float | Iterable[float],
    ek: float,
    mode_tables: dict[str, tuple[np.ndarray, np.ndarray]],
    modes: Sequence[str] = ("000", "penu", "penug", "pm0", "pm0g", "pmunu", "pmunug"),
) -> float | np.ndarray:
    bitflags = mode_bitflags(modes)
    if hasattr(egam, "__len__"):
        energies = np.asarray(list(egam), dtype=np.float64)
        if energies.ndim != 1:
            raise ValueError("Photon energies must be 0 or 1-dimensional.")
        return np.array(
            [long_kaon_decay_spectrum_point(float(e), ek, bitflags, mode_tables) for e in energies],
            dtype=np.float64,
        )
    return long_kaon_decay_spectrum_point(float(egam), ek, bitflags, mode_tables)
