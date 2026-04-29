"""Polya-Gamma omega updates for latent space model likelihood."""

from __future__ import annotations

import math


def update_omega_single(xit, xit_sigma, xjt, xjt_sigma, deltai, deltai_sigma, deltaj, deltaj_sigma):
    psi_sq = 0.0
    psi_sq += deltai_sigma + deltai * deltai
    psi_sq += deltaj_sigma + deltaj * deltaj
    psi_sq += 2.0 * deltai * deltaj
    n_features = len(xit)
    for p in range(n_features):
        psi_sq += 2.0 * (deltai + deltaj) * xit[p] * xjt[p]
        for q in range(n_features):
            psi_sq += (xit_sigma[p][q] + xit[p] * xit[q]) * (xjt_sigma[p][q] + xjt[p] * xjt[q])

    c_omega = math.sqrt(psi_sq)
    omega = math.tanh(0.5 * c_omega) / (2.0 * c_omega)
    return omega, psi_sq


def update_omega(y, omega, x, x_sigma, delta, delta_sigma):
    n_time = len(omega)
    n_nodes = len(omega[0])
    n_features = len(x[0][0])
    loglik = 0.0
    for t in range(n_time):
        for i in range(n_nodes):
            for j in range(i):
                if y[t][i][j] != -1.0:
                    om, psi_sq = update_omega_single(
                        x[t][i],
                        x_sigma[t][i],
                        x[t][j],
                        x_sigma[t][j],
                        delta[i],
                        delta_sigma[i],
                        delta[j],
                        delta_sigma[j],
                    )
                    omega[t][i][j] = om
                    omega[t][j][i] = om
                    psi = delta[i] + delta[j]
                    for p in range(n_features):
                        psi += x[t][i][p] * x[t][j][p]
                    loglik += (y[t][i][j] - 0.5) * psi - 0.5 * om * psi_sq
    return loglik
