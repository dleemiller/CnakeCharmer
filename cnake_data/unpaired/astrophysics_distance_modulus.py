from __future__ import annotations

import math


def distance_modulus(distance_mpc: float) -> float:
    if distance_mpc <= 0:
        raise ValueError("distance must be positive")
    d_pc = distance_mpc * 1_000_000.0
    return 5.0 * math.log10(d_pc) - 5.0


def absolute_magnitude(apparent_mag: float, distance_mpc: float) -> float:
    return apparent_mag - distance_modulus(distance_mpc)
