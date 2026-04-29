"""Mixture-normal forecast density helpers."""

from __future__ import annotations

import numpy as np


def expit(z):
    return 1.0 / (1.0 + np.exp(-z))


def normal_pdf(x, mean, var):
    n_features = x.shape[0]
    sum_sq = np.sum((x - mean) ** 2) * 0.5 * (1.0 / var)
    return np.exp(-0.5 * n_features * np.log(2 * np.pi * var) - sum_sq)


def mixture_normal_pdf(x, x_prev, weights, lmbda, mean, sigma):
    n_groups = mean.shape[0]
    res = 0.0
    for k in range(n_groups):
        mu = lmbda * mean[k] + (1.0 - lmbda) * x_prev
        res += weights[k] * normal_pdf(x, mu, sigma[k])
    return res


def renormalize_weights(z, weights, means, sigmas):
    active_groups, znew = np.unique(z, return_inverse=True)
    trans_w = weights[active_groups][:, active_groups]
    trans_w /= np.sum(trans_w, axis=1).reshape(-1, 1)
    mu = means[active_groups]
    sigma = sigmas[active_groups]
    return znew, trans_w, mu, sigma
