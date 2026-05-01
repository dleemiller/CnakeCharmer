"""Copula simulation loops and a simple DF grid-search likelihood score."""

from __future__ import annotations

import numpy as np


def unif_from_gauss_copula(copula_sampler, log_return_corr, num_iterations):
    out = np.zeros((log_return_corr.shape[0], num_iterations), dtype=float)
    for i in range(num_iterations):
        sample = copula_sampler(log_return_corr)
        for j in range(len(sample)):
            out[j, i] = sample[j]
    return out


def unif_from_t_copula(copula_sampler, rank_corr, t_df, num_iterations):
    out = np.zeros((rank_corr.shape[0], num_iterations), dtype=float)
    for i in range(num_iterations):
        sample = copula_sampler(rank_corr, t_df)
        for j in range(len(sample)):
            out[j, i] = sample[j]
    return out


def t_copula_df_mle(u_hist_t, density_fn, min_df=1, max_df=23):
    best_df = min_df
    best_score = float("-inf")
    scores = np.zeros(max_df + 1, dtype=float)

    for df in range(min_df, max_df + 1):
        total = 0.0
        for i in range(u_hist_t.shape[0]):
            density = density_fn(u_hist_t[i], df)
            total += np.log(max(density, 1e-300))
        scores[df] = total
        if total > best_score:
            best_score = total
            best_df = df

    scores[0] = best_df
    return scores
