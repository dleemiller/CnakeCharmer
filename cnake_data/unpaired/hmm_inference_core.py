"""Forward log-partition and gradient helpers for HMMs."""

from __future__ import annotations

import math


def normalize_inplace(a):
    tot = sum(a)
    for i in range(len(a)):
        a[i] /= tot
    return tot


def logsumexp(a):
    cmax = max(a)
    tot = 0.0
    for v in a:
        tot += math.exp(v - cmax)
    return math.log(tot) + cmax


def hmm_logz(init_params, pair_params, node_params):
    t_len = len(node_params)
    n_states = len(node_params[0])

    log_alpha = [[0.0] * n_states for _ in range(t_len)]
    for i in range(n_states):
        log_alpha[0][i] = init_params[i] + node_params[0][i]

    for t in range(1, t_len):
        for i in range(n_states):
            tmp = [log_alpha[t - 1][j] + pair_params[j][i] for j in range(n_states)]
            log_alpha[t][i] = logsumexp(tmp) + node_params[t][i]

    return logsumexp(log_alpha[-1]), log_alpha
