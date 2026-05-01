from __future__ import annotations

import math

ATM_M = 0.029
R = 8.3144598
MEAN_CS = 0.866
CONVERSION_RATIO = 8.0


def find_pressure(h: float, pb: float, tb: float, g: float) -> float:
    return pb * math.exp((-g * ATM_M * h) / (R * tb))


def find_cs_ratio(lat: float) -> float:
    return math.cos(lat) / MEAN_CS


def t_from_stored_v(stored_v: int) -> float:
    return float(stored_v) / CONVERSION_RATIO


def stored_v_from_t(t: float) -> int:
    return int(t * CONVERSION_RATIO)
