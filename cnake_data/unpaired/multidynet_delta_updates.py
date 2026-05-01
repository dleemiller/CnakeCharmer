"""Natural-parameter accumulation and mean-field delta updates."""

from __future__ import annotations


def calculate_natural_parameters(Y, XLX, delta, omega, k, i):
    n_nodes = Y.shape[1]
    eta1 = 0.0
    eta2 = 0.0
    for j in range(n_nodes):
        if j != i and Y[k, i, j] != -1.0:
            eta1 += Y[k, i, j] - 0.5 - omega[k, i, j] * (delta[k, j] + XLX[k, i, j])
            eta2 += omega[k, i, j]
    return eta1, eta2


def update_deltas(Y, delta, delta_sigma, XLX, omega, tau_prec):
    n_layers = Y.shape[0]
    n_nodes = Y.shape[1]
    for k in range(n_layers):
        for i in range(n_nodes):
            A, B = calculate_natural_parameters(Y, XLX, delta, omega, k, i)
            delta_sigma[k, i] = 1.0 / (B + tau_prec)
            delta[k, i] = delta_sigma[k, i] * A
