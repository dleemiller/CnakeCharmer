from __future__ import annotations

import math


def decode_vector_components(
    x: float, y: float, normal_mode: bool, sign_bit: int = 0, z_raw: float = 0.0
) -> tuple[float, float, float]:
    if normal_mode:
        f = x * x + y * y
        z = 0.0 if f <= 1.0 else math.sqrt(1.0 - f)
        if sign_bit:
            z *= -1.0
    else:
        z = z_raw
    return x, y, z
