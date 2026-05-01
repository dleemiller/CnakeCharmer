"""Diagonal signature-kernel dynamic programming approximations."""

from __future__ import annotations

import numpy as np


def sig_kern_diag(x: np.ndarray, n: int = 0, solver: int = 1):
    A, M, D = x.shape
    N = M
    factor = 2 ** (2 * n)
    MM = (2**n) * (M - 1)
    NN = (2**n) * (N - 1)
    K = np.zeros((A, MM + 1, NN + 1), dtype=np.float64)
    K_rev = np.zeros((A, MM + 1, NN + 1), dtype=np.float64)

    for l in range(A):
        K[l, :, 0] = 1.0
        K_rev[l, :, 0] = 1.0

        for i in range(MM):
            for j in range(i):
                ii = int(i / (2**n))
                jj = int(j / (2**n))
                increment = 0.0
                increment_rev = 0.0
                for k in range(D):
                    increment += (
                        (x[l, ii + 1, k] - x[l, ii, k]) * (x[l, jj + 1, k] - x[l, jj, k]) / factor
                    )
                    increment_rev += (
                        (x[l, (M - 1) - (ii + 1), k] - x[l, (M - 1) - ii, k])
                        * (x[l, (N - 1) - (jj + 1), k] - x[l, (N - 1) - jj, k])
                        / factor
                    )

                if solver == 0:
                    K[l, i + 1, j + 1] = (K[l, i, j + 1] + K[l, i + 1, j]) + K[l, i, j] * (
                        increment - 1.0
                    )
                else:
                    K[l, i + 1, j + 1] = (K[l, i, j + 1] + K[l, i + 1, j]) * (
                        1.0 + 0.5 * increment + (1.0 / 12.0) * increment**2
                    ) - K[l, i, j] * (1.0 - (1.0 / 12.0) * increment**2)

                K_rev[l, i + 1, j + 1] = (K_rev[l, i, j + 1] + K_rev[l, i + 1, j]) + K_rev[
                    l, i, j
                ] * (increment_rev - 1.0)

            ii = int(i / (2**n))
            jj = int(i / (2**n))
            increment = 0.0
            increment_rev = 0.0
            for k in range(D):
                increment += (
                    (x[l, ii + 1, k] - x[l, ii, k]) * (x[l, jj + 1, k] - x[l, jj, k]) / factor
                )
                increment_rev += (
                    (x[l, (M - 1) - (ii + 1), k] - x[l, (M - 1) - ii, k])
                    * (x[l, (N - 1) - (jj + 1), k] - x[l, (N - 1) - jj, k])
                    / factor
                )

            K[l, i + 1, i + 1] = 2 * K[l, i + 1, i] * (
                1.0 + 0.5 * increment + (1.0 / 12.0) * increment**2
            ) - K[l, i, i] * (1.0 - (1.0 / 12.0) * increment**2)
            K_rev[l, i + 1, i + 1] = 2 * K_rev[l, i + 1, i] + K_rev[l, i, i] * (increment_rev - 1.0)

    return np.array(K), np.array(K_rev)
