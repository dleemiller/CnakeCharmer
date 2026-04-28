"""Variational Bayes helper kernels for topic models."""

from __future__ import annotations

import math

import numpy as np


def calculate_doc_post_sufficient_stat(
    n_k: np.ndarray,
    a_k: np.ndarray,
    X: np.ndarray,
    n_topics: int,
    n_samples: int,
    n_features: int,
    exp_doc_topic: np.ndarray,
    exp_topic_word_distr: np.ndarray,
) -> None:
    n_k[:] = 0.0
    for dd in range(n_samples):
        for word_id in range(n_features):
            x = X[dd, word_id]
            if x == 0:
                continue
            norm = 1.0e-10
            for k in range(n_topics):
                norm += exp_doc_topic[dd, k] * a_k[k] * exp_topic_word_distr[k, word_id]
            for k in range(n_topics):
                n_k[k] += (
                    x * exp_doc_topic[dd, k] * a_k[k] * exp_topic_word_distr[k, word_id] / norm
                )


def loglikelihood(n_k: np.ndarray) -> float:
    total = 0
    likelihood = 0.0
    for val in n_k:
        if val >= 1:
            likelihood -= float(val) * math.log(float(val))
            total += int(val)
    if total >= 1:
        return float(total * math.log(total) + likelihood)
    return 0.0


def mdl_sparse_differential(
    n_k: np.ndarray, c_mn_diff: np.ndarray, V: int, a_k: np.ndarray
) -> None:
    bound = c_mn_diff.shape[0]
    for k in range(n_k.shape[0]):
        if n_k[k] < bound:
            a_k[k] = math.exp(-c_mn_diff[int(n_k[k])])
        else:
            a_k[k] = math.exp(-(V - 1) / 2.0 / n_k[k])


def mean_change(arr_1: np.ndarray, arr_2: np.ndarray) -> float:
    return float(np.abs(arr_1 - arr_2).mean())


def psi(x: float) -> float:
    euler = 0.5772156649015329
    if x <= 1e-6:
        return -euler - 1.0 / x
    result = 0.0
    while x < 6.0:
        result -= 1.0 / x
        x += 1.0
    r = 1.0 / x
    result += math.log(x) - 0.5 * r
    r = r * r
    result -= r * ((1.0 / 12.0) - r * ((1.0 / 120.0) - r * (1.0 / 252.0)))
    return result


def dirichlet_expectation_1d(
    doc_topic: np.ndarray, doc_topic_prior: np.ndarray, out: np.ndarray
) -> None:
    total = 0.0
    for i in range(doc_topic.shape[0]):
        dt = doc_topic[i] + doc_topic_prior[i]
        doc_topic[i] = dt
        total += dt
    psi_total = psi(total)
    for i in range(doc_topic.shape[0]):
        out[i] = math.exp(psi(doc_topic[i]) - psi_total)


def dirichlet_expectation_2d(arr: np.ndarray) -> np.ndarray:
    d_exp = np.empty_like(arr)
    for i in range(arr.shape[0]):
        row_total = float(np.sum(arr[i]))
        psi_row_total = psi(row_total)
        for j in range(arr.shape[1]):
            d_exp[i, j] = psi(float(arr[i, j])) - psi_row_total
    return d_exp
