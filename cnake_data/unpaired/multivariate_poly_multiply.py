"""Multiply two sets of multivariate polynomial terms."""

from __future__ import annotations

import numpy as np


def multiply_multivariate_polynomials(indices_i, coeffs_i, indices_ii, coeffs_ii):
    indices_i = np.asarray(indices_i)
    coeffs_i = np.asarray(coeffs_i, dtype=float)
    indices_ii = np.asarray(indices_ii)
    coeffs_ii = np.asarray(coeffs_ii, dtype=float)

    num_vars = indices_i.shape[0]
    num_i = indices_i.shape[1]
    num_ii = indices_ii.shape[1]

    out_indices = np.empty((num_vars, num_i * num_ii), dtype=indices_i.dtype)
    out_coeffs = np.empty((num_i * num_ii,), dtype=float)

    kk = 0
    for ii in range(num_i):
        index1 = indices_i[:, ii]
        for jj in range(num_ii):
            out_indices[:, kk] = index1 + indices_ii[:, jj]
            out_coeffs[kk] = coeffs_i[ii] * coeffs_ii[jj]
            kk += 1

    return out_indices, out_coeffs
