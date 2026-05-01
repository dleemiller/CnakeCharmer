"""Null/NaN/Inf object scanner."""

from __future__ import annotations

import numpy as np

INF = float(np.inf)
NEGINF = -INF


def checknull(val):
    if isinstance(val, float):
        return val != val or val == INF or val == NEGINF
    return val is None


def isnullobj(arr):
    flat = np.ravel(arr)
    out = np.zeros(flat.shape[0], dtype=np.int8)
    for i, val in enumerate(flat):
        if checknull(val):
            out[i] = 1
    return out
