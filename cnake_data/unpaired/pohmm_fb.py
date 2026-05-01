"""Forward/backward and eta computation in log-space HMM."""

from __future__ import annotations

import numpy as np

NINF = -np.inf


def _max(values):
    vmax = NINF
    for value in values:
        if value > vmax:
            vmax = value
    return vmax


def _logsumexp(x):
    vmax = _max(x)
    power_sum = 0.0
    for xi in x:
        power_sum += np.exp(xi - vmax)
    return np.log(power_sum) + vmax


def forward(n_observations, n_substates, event_idx, log_startprob, log_transmat, framelogprob):
    fwdlattice = np.zeros((n_observations, n_substates), dtype=float)
    work_buffer = np.zeros(n_substates, dtype=float)

    for i in range(n_substates):
        fwdlattice[0, i] = log_startprob[event_idx[0]][i] + framelogprob[0, i]

    for t in range(1, n_observations):
        for j in range(n_substates):
            for i in range(n_substates):
                work_buffer[i] = (
                    fwdlattice[t - 1, i] + log_transmat[event_idx[t - 1], event_idx[t], i, j]
                )
            fwdlattice[t, j] = _logsumexp(work_buffer) + framelogprob[t, j]

    return fwdlattice


def backward(n_observations, n_substates, event_idx, log_transmat, framelogprob):
    bwdlattice = np.zeros((n_observations, n_substates), dtype=float)
    work_buffer = np.zeros(n_substates, dtype=float)

    for t in range(n_observations - 2, -1, -1):
        for i in range(n_substates):
            for j in range(n_substates):
                work_buffer[j] = (
                    log_transmat[event_idx[t], event_idx[t + 1], i, j]
                    + framelogprob[t + 1, j]
                    + bwdlattice[t + 1, j]
                )
            bwdlattice[t, i] = _logsumexp(work_buffer)
    return bwdlattice


def compute_lneta(
    n_observations, n_substates, event_idx, fwdlattice, log_transmat, bwdlattice, framelogprob
):
    lneta = np.zeros((n_observations - 1, n_substates, n_substates), dtype=float)
    logprob = _logsumexp(fwdlattice[-1])
    for t in range(n_observations - 1):
        for i in range(n_substates):
            for j in range(n_substates):
                lneta[t, i, j] = (
                    fwdlattice[t, i]
                    + log_transmat[event_idx[t], event_idx[t + 1], i, j]
                    + framelogprob[t + 1, j]
                    + bwdlattice[t + 1, j]
                    - logprob
                )
    return lneta
