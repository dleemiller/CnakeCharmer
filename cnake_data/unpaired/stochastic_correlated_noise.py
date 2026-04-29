"""Correlated complex noise generation from target correlation."""

from __future__ import annotations

import numpy as np


def cor(lam, g, a, b, t):
    gam = a
    temp = b
    re = 2.0 * lam * temp * np.exp(-gam * np.abs(t))
    im = -lam * gam * np.exp(-gam * np.abs(t))
    return re + 1j * im


def run(i_tmax=800, dt=1.0, n_samples=1000):
    g11 = 1.0 / 200
    l11 = 1.0
    a = 1.0 / 100.0
    b = a / 3.0

    mean = np.zeros(2 * i_tmax, dtype=np.float64)
    cov = np.zeros((2 * i_tmax, 2 * i_tmax), dtype=np.float64)
    val = np.zeros(2 * i_tmax, dtype=np.complex128)

    for itau in range(2 * i_tmax):
        val[itau] = cor(l11, g11, a, b, itau * dt - i_tmax)

    for it1 in range(i_tmax):
        for it2 in range(i_tmax):
            itau = it1 + i_tmax - it2
            cov[it1, it2] = 3.0 * np.real(val[itau])
            cov[i_tmax + it1, it2] = np.imag(val[itau])
            cov[it1, i_tmax + it2] = np.imag(val[itau])
            cov[i_tmax + it1, i_tmax + it2] = np.real(val[itau])

    cov = cov / 2.0
    xi = np.random.multivariate_normal(mean, cov, n_samples).T

    zi = np.zeros(xi.shape, dtype=np.complex128)
    for i in range(i_tmax):
        zi[i, :] = xi[i, :] + 1j * xi[i_tmax + i, :]

    ct = np.zeros(i_tmax, dtype=np.complex128)
    for it in range(i_tmax):
        ct[it] = np.sum(zi[0, :] * zi[it, :]) / xi.shape[1]

    return zi, ct
