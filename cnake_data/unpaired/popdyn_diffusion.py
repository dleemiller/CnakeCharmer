"""Population dynamics and bounded 1D diffusion stepper."""

from __future__ import annotations

import numpy as np


def popdyn(l, dts, observations, alpha, beta, tau):
    p = np.empty((len(observations), l), dtype=float)
    n = np.ones(l, dtype=float)
    edts = np.exp(-dts / tau)

    for i in range(dts.shape[0]):
        nsum = 0.0
        for j in range(n.shape[0]):
            n[j] = 1.0 + (n[j] - 1.0) * edts[i]
            nsum += n[j]
        for j in range(n.shape[0]):
            p[i, j] = n[j] / nsum
        obs = observations[i]
        n[obs] = alpha + (1.0 + beta) * n[obs]
    return p


def _clip(x, xmin=0.0, xmax=1.0):
    if x < xmin:
        return xmin
    if x > xmax:
        return xmax
    return x


def step1ddiffusion(q, dt, alpha, beta, dtmax, prng=np.random):
    if dt < dtmax:
        dw = prng.normal()
        q += dt * 0.5 * (alpha - (alpha + beta) * q) + (q * (1.0 - q) * dt) ** 0.5 * dw
        return _clip(q, 0, 1)

    nsteps = int(dt / dtmax) + 1
    dt /= nsteps
    rand = prng.normal(size=nsteps)
    for i in range(nsteps):
        q += dt * 0.5 * (alpha - (alpha + beta) * q) + (q * (1.0 - q) * dt) ** 0.5 * rand[i]
        q = _clip(q, 0, 1)
    return q
