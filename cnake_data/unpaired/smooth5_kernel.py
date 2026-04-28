"""Five-point smoothing for integer and float rasters."""

from __future__ import annotations

import numpy as np


def _round_int(a: float) -> int:
    if a % 1 < 0.5:
        return int(a // 1)
    return int(a // 1) + 1


def _smooth5_int(img: np.ndarray, nodata: int) -> np.ndarray:
    ny, nx = img.shape
    out = np.zeros((ny, nx), dtype=np.int16)
    for i in range(1, ny - 1):
        for j in range(1, nx - 1):
            runsum = 0.0
            count = 0
            if img[i, j] != nodata:
                vals = [img[i, j], img[i - 1, j], img[i, j - 1], img[i, j + 1], img[i + 1, j]]
                for v in vals:
                    if v != nodata:
                        runsum += float(v)
                        count += 1
            out[i, j] = nodata if count == 0 else _round_int(runsum / count)
    out[0, :] = nodata
    out[-1, :] = nodata
    out[:, 0] = nodata
    out[:, -1] = nodata
    return out


def _smooth5_double(img: np.ndarray) -> np.ndarray:
    ny, nx = img.shape
    out = np.zeros((ny, nx), dtype=np.float64)
    for i in range(1, ny - 1):
        for j in range(1, nx - 1):
            runsum = 0.0
            count = 0
            vals = [img[i, j], img[i - 1, j], img[i, j - 1], img[i, j + 1], img[i + 1, j]]
            for v in vals:
                if not np.isnan(v):
                    runsum += float(v)
                    count += 1
            out[i, j] = np.nan if count == 0 else runsum / count
    out[0, :] = np.nan
    out[-1, :] = np.nan
    out[:, 0] = np.nan
    out[:, -1] = np.nan
    return out


def smooth5_int(img: np.ndarray, niter: int = 1, nodata: int = -1) -> np.ndarray:
    out = np.asarray(img, dtype=np.int16)
    for _ in range(niter):
        out = _smooth5_int(out, nodata)
    return out


def smooth5(img: np.ndarray, niter: int = 1) -> np.ndarray:
    out = np.asarray(img, dtype=np.float64)
    for _ in range(niter):
        out = _smooth5_double(out)
    return out
