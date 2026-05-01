"""Liability-threshold helper kernels."""

from __future__ import annotations

import numpy as np

CASE = 2
CONTROL = 1


def standardize_vector(x):
    x = np.asarray(x, dtype=float)
    var_x = np.var(x)
    if var_x == 0:
        raise ValueError("vector has zero variance")
    return (x - np.mean(x)) / np.sqrt(var_x)


def calc_he_regression_herit(grm, pheno_norm):
    grm = np.asarray(grm, dtype=float)
    pheno_norm = np.asarray(pheno_norm, dtype=float)
    n = grm.shape[0]

    h2_num = 0.0
    h2_denom = 0.0
    for i in range(n):
        for j in range(n):
            if i != j:
                h2_num += grm[i, j] * pheno_norm[i] * pheno_norm[j]
                h2_denom += grm[i, j] * grm[i, j]
    return h2_num / h2_denom if h2_denom != 0 else 0.0
