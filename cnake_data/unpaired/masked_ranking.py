"""Masked ranking utilities for 2D arrays."""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def is_missing(data: np.ndarray, missing_value) -> np.ndarray:
    if np.issubdtype(data.dtype, np.floating) and np.isnan(missing_value):
        return np.isnan(data)
    return data == missing_value


def rankdata_1d_descending(data: np.ndarray, method: str) -> np.ndarray:
    return rankdata(-(data.view(np.float64)), method=method)


def rankdata_2d_ordinal(array: np.ndarray) -> np.ndarray:
    nrows, ncols = array.shape
    sort_idxs = np.argsort(array, axis=1, kind="mergesort")
    out = np.empty_like(array, dtype=np.float64)
    for i in range(nrows):
        for j in range(ncols):
            out[i, sort_idxs[i, j]] = j + 1.0
    return out


def masked_rankdata_2d(
    data: np.ndarray, mask: np.ndarray, missing_value, method: str, ascending: bool
) -> np.ndarray:
    if data.dtype.name not in ("float64", "int64", "datetime64[ns]"):
        raise TypeError(f"Can't compute rankdata on array of dtype {data.dtype.name!r}.")

    missing_locations = (~mask) | is_missing(data, missing_value)
    values = data.copy().view(np.float64)
    values[missing_locations] = np.nan
    if not ascending:
        values = -values

    if method == "ordinal":
        result = rankdata_2d_ordinal(values)
    else:
        result = np.apply_along_axis(rankdata, 1, values, method=method)
        if result.dtype.name != "float64":
            result = result.astype("float64")

    result[missing_locations] = np.nan
    return result
