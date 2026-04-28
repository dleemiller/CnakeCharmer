"""Row-wise gini impurity aggregation kernel."""

from __future__ import annotations

import numpy as np


def gini_impurity(distribution: np.ndarray) -> float:
    arr = np.asarray(distribution)
    if arr.ndim != 2:
        raise ValueError("distribution must be 2D")
    nrow, ncol = arr.shape
    if nrow == 0:
        return 0.0

    result = 0.0
    for i in range(nrow):
        row = arr[i]
        group_count = 0
        count2 = 0
        for j in range(ncol):
            x = int(row[j])
            group_count += x
            count2 += x * x
        if group_count > 0:
            result += ((group_count * group_count) - count2) / group_count
    return float(result / nrow)
