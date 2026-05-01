"""Polygon mask coverage and valid-area metrics aggregation."""

from __future__ import annotations

import numpy as np


def cal_area(
    data: np.ndarray, mask: np.ndarray, fid: np.ndarray, nodata: np.ndarray, nthreads=None
):
    var_size = data.shape[0]
    poly_num = fid.shape[0]
    row, col = data.shape[1], data.shape[2]

    results = np.zeros((poly_num, var_size), dtype=np.float64)
    vfid = np.zeros(poly_num, dtype=np.int64)

    poly_area = np.zeros(poly_num, dtype=np.float64)
    valid_area = np.zeros(poly_num, dtype=np.float64)

    for i in range(poly_num):
        for p in range(row):
            for q in range(col):
                if mask[p, q] != fid[i]:
                    continue
                poly_area[i] += 1
                if abs(data[3, p, q] - nodata[3]) > 1e-14:
                    valid_area[i] += 1
                    if data[3, p, q] >= -350.0:
                        results[i, 3] += 1
                    else:
                        for j in range(var_size - 2):
                            if abs(data[j, p, q] - nodata[j]) > 1e-14:
                                results[i, j] += data[j, p, q] / 100.0
                if abs(data[4, p, q] - nodata[4]) > 1e-14:
                    results[i, 4] += 1
                    valid_area[i] += 1

        if poly_area[i] < 1e-14:
            vfid[i] = -1
        elif valid_area[i] / poly_area[i] > 0.9:
            for j in range(var_size):
                results[i, j] = results[i, j] / valid_area[i]
            vfid[i] = fid[i]
        else:
            vfid[i] = -1

    return results, vfid
