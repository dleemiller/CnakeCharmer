"""Map feature matrix values to binned integer levels."""

from __future__ import annotations

import math

import numpy as np


def _map_col_to_bins(data, binning_thresholds, is_categorical, missing_values_bin_idx):
    binned = np.empty(data.shape[0], dtype=np.int64)
    for i in range(data.shape[0]):
        val = data[i]
        if math.isnan(val) or (is_categorical and val < 0):
            binned[i] = missing_values_bin_idx
        else:
            left, right = 0, binning_thresholds.shape[0]
            while left < right:
                middle = left + (right - left - 1) // 2
                if val <= binning_thresholds[middle]:
                    right = middle
                else:
                    left = middle + 1
            binned[i] = left
    return binned


def map_to_bins(data, binning_thresholds, is_categorical, missing_values_bin_idx):
    data = np.asarray(data)
    n_samples, n_features = data.shape
    binned = np.empty((n_samples, n_features), dtype=np.int64, order="F")
    for feature_idx in range(n_features):
        binned[:, feature_idx] = _map_col_to_bins(
            data[:, feature_idx],
            np.asarray(binning_thresholds[feature_idx]),
            bool(is_categorical[feature_idx]),
            missing_values_bin_idx,
        )
    return binned
