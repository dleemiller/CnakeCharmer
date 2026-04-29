"""Timedelta component extraction on int64 nanosecond matrices."""

from __future__ import annotations

import math

NAT = -(2**63)


def seconds(a):
    nr = len(a)
    nc = len(a[0]) if nr else 0
    out = [[math.nan] * nc for _ in range(nr)]
    for j in range(nc):
        for i in range(nr):
            v = a[i][j]
            if v != NAT:
                out[i][j] = int(v / 10**9) % 86400
    return out


def milliseconds(a, total=False):
    nr = len(a)
    nc = len(a[0]) if nr else 0
    out = [[math.nan] * nc for _ in range(nr)]
    for j in range(nc):
        for i in range(nr):
            v = a[i][j]
            if v == NAT:
                continue
            frac = (v % 10**9) / 10**6
            out[i][j] = frac if total else int(frac)
    return out


def days(a, total=False):
    nr = len(a)
    nc = len(a[0]) if nr else 0
    out = [[math.nan] * nc for _ in range(nr)]
    nanos = 10**9 * 86400
    for j in range(nc):
        for i in range(nr):
            v = a[i][j]
            if v == NAT:
                continue
            d = v / nanos
            out[i][j] = d if total else int(d)
    return out
