"""Matrix transpose kernel."""

from __future__ import annotations

import numpy as np


def transpose(A):
    arr = np.asarray(A, dtype=np.float64)
    m_a, n_a = arr.shape
    B = np.empty((n_a, m_a), dtype=np.float64)
    for i in range(m_a):
        for j in range(n_a):
            B[j, i] = arr[i, j]
    return B
