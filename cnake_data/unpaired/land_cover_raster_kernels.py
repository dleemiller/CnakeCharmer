from __future__ import annotations

import math

MISSING_VALUE = -32768
UNKNOWN_VALUE = -9999


def convert_evt_classes(arr: list[list[float]], evt_to_class: list[int]) -> None:
    w = len(arr)
    for i in range(w):
        for j in range(w):
            arr[i][j] = float(evt_to_class[int(arr[i][j]) - 3000])


def one_hot_encode(arr: list[list[float]], num_classes: int) -> list[list[list[float]]]:
    w = len(arr)
    out = [[[0.0 for _ in range(w)] for _ in range(w)] for _ in range(num_classes)]
    for i in range(w):
        for j in range(w):
            c = int(arr[i][j])
            for k in range(num_classes):
                out[k][i][j] = 1.0 if c == k else 0.0
    return out


def process_raster_inplace(raster: list[list[float]]) -> None:
    w = len(raster)
    for i in range(w):
        for j in range(w):
            v = raster[i][j]
            if v in (MISSING_VALUE, UNKNOWN_VALUE):
                raster[i][j] = math.nan
