"""Invert dense flow by splatting strongest-magnitude vectors to destination grid."""

from __future__ import annotations

import numpy as np


def _nearest_resize(arr: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    src_h, src_w = arr.shape[:2]
    y_idx = np.clip((np.arange(out_h) * src_h / out_h).astype(int), 0, src_h - 1)
    x_idx = np.clip((np.arange(out_w) * src_w / out_w).astype(int), 0, src_w - 1)
    if arr.ndim == 2:
        return arr[y_idx][:, x_idx]
    return arr[y_idx][:, x_idx, :]


def invert_flow(flow: np.ndarray, mag: np.ndarray, fac: float) -> tuple[np.ndarray, np.ndarray]:
    cols, rows = flow.shape[0], flow.shape[1]
    factor = float(fac)
    colsf = int(cols * factor)
    rowsf = int(rows * factor)

    new_mag = np.zeros((colsf, rowsf), dtype=np.float32)
    new_mask = np.full((colsf, rowsf), 255, dtype=np.uint8)
    new_flow = np.zeros((colsf, rowsf, 2), dtype=np.int32)

    for i in range(cols):
        for j in range(rows):
            i2 = int((i + flow[i, j, 1]) * factor)
            if i2 < 0 or i2 > colsf - 1:
                continue
            j2 = int((j + flow[i, j, 0]) * factor)
            if j2 < 0 or j2 > rowsf - 1:
                continue
            if mag[i, j] < new_mag[i2, j2]:
                continue
            new_mag[i2, j2] = mag[i, j]
            new_flow[i2, j2, :] = flow[i, j, :]
            new_mask[i2, j2] = 0

    out_flow = _nearest_resize(new_flow.astype(np.float32), cols, rows)
    out_mask = _nearest_resize(new_mask, cols, rows)
    return out_flow, out_mask
