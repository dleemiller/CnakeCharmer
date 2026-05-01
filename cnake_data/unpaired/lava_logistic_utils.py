"""Logistic objective helpers with LAVA-style shrinkage."""

from __future__ import annotations

import math


def recover_delta(beta, alpha_1, alpha_2):
    out = [0.0] * len(beta)
    threshold = (alpha_1 / alpha_2) / 2.0
    for i, b in enumerate(beta):
        if b > threshold:
            out[i] = b - threshold
        elif b < -threshold:
            out[i] = b + threshold
        else:
            out[i] = 0.0
    return out


def logistic_loss_grad(beta, x, y):
    n = len(x)
    p = len(beta)
    g = [0.0] * p
    loss = 0.0
    for i in range(n):
        xb = sum(x[i][j] * beta[j] for j in range(p))
        prob = 1.0 / (1.0 + math.exp(-xb))
        loss -= -math.log(1.0 + math.exp(xb)) + y[i] * xb
        for j in range(p):
            g[j] -= ((prob - 1.0 + y[i]) * x[i][j]) / n
    return loss / n, g
