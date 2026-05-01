from __future__ import annotations

import math


def _logsumexp(xs: list[float]) -> float:
    m = max(xs)
    if math.isinf(m):
        return float("-inf")
    acc = 0.0
    for x in xs:
        acc += math.exp(x - m)
    return math.log(acc) + m


def forward_log(
    log_start: list[float], log_trans: list[list[float]], frame_logprob: list[list[float]]
) -> list[list[float]]:
    n_samples = len(frame_logprob)
    n_components = len(log_start)
    fwd = [[0.0 for _ in range(n_components)] for _ in range(n_samples)]
    for i in range(n_components):
        fwd[0][i] = log_start[i] + frame_logprob[0][i]
    for t in range(1, n_samples):
        for j in range(n_components):
            work = [fwd[t - 1][i] + log_trans[i][j] for i in range(n_components)]
            fwd[t][j] = _logsumexp(work) + frame_logprob[t][j]
    return fwd


def viterbi_log(
    log_start: list[float], log_trans: list[list[float]], frame_logprob: list[list[float]]
) -> tuple[list[int], float]:
    n_samples = len(frame_logprob)
    n_components = len(log_start)
    vit = [[0.0 for _ in range(n_components)] for _ in range(n_samples)]
    back = [[0 for _ in range(n_components)] for _ in range(n_samples)]
    for i in range(n_components):
        vit[0][i] = log_start[i] + frame_logprob[0][i]
    for t in range(1, n_samples):
        for i in range(n_components):
            cand = [vit[t - 1][j] + log_trans[j][i] for j in range(n_components)]
            best_j = max(range(n_components), key=lambda j: cand[j])
            vit[t][i] = cand[best_j] + frame_logprob[t][i]
            back[t][i] = best_j
    last = max(range(n_components), key=lambda i: vit[n_samples - 1][i])
    path = [0] * n_samples
    path[-1] = last
    for t in range(n_samples - 2, -1, -1):
        path[t] = back[t + 1][path[t + 1]]
    return path, vit[n_samples - 1][last]
