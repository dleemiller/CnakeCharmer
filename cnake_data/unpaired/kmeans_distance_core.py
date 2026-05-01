from __future__ import annotations

import math


def sq_dist(x: list[float], y: list[float]) -> float:
    if len(x) != len(y) or not x:
        return -1.0
    s = 0.0
    for i in range(len(x)):
        d = x[i] - y[i]
        s += d * d
    return s


def sq_distances(data: list[list[float]], centers: list[list[float]]) -> list[list[float]]:
    out = [[0.0 for _ in range(len(centers))] for _ in range(len(data))]
    for i in range(len(data)):
        for j in range(len(centers)):
            out[i][j] = sq_dist(data[i], centers[j])
    return out


def distances(data: list[list[float]], centers: list[list[float]]) -> list[list[float]]:
    d2 = sq_distances(data, centers)
    for i in range(len(d2)):
        for j in range(len(d2[i])):
            d2[i][j] = math.sqrt(d2[i][j])
    return d2
