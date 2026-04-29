"""Hourly/daily clearness-index helpers."""

from __future__ import annotations

import math

import numpy as np


def kt_h(gsc, lat, day_of_year, hour, eccentricity, sunrise_h, global_h):
    if global_h <= 0:
        return np.nan

    hour = hour + 0.5
    sdec = math.radians(23.45) * math.sin(math.radians(360 * (284 + day_of_year) / 365))
    ws = -math.acos(-(math.tan(lat) * math.tan(sdec)))
    w = ws + (hour - sunrise_h) * math.radians(15)
    go = (
        gsc
        * eccentricity
        * (math.cos(lat) * math.cos(sdec) * math.cos(w) + math.sin(lat) * math.sin(sdec))
    )

    if go == 0:
        return np.nan
    return max(0.0, min(global_h / go, 1.0))


def apply_kt_h(gsc, lat, col_n, col_h, col_e0, col_sunrise_h, col_global_h):
    out = np.empty(len(col_h), dtype=float)
    for i in range(len(col_h)):
        out[i] = kt_h(gsc, lat, col_n[i], col_h[i], col_e0[i], col_sunrise_h[i], col_global_h[i])
    return out
