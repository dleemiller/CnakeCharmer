"""Coordinate-descent update kernel for constrained NMF."""

from __future__ import annotations


def update_cdnmf_fast(w, w_orig, hht, xht, permutation):
    violation = 0.0
    n_components = len(w[0])
    n_samples = len(w)

    for s in range(n_components):
        t = permutation[s]
        for i in range(n_samples):
            grad = -xht[i][t]
            for r in range(n_components):
                grad += hht[t][r] * w[i][r]

            if (w[i][t] == w_orig[i][t] and grad < 0.0) or (w[i][t] == 0.0 and grad > 0.0):
                pg = 0.0
            else:
                pg = grad
            violation += abs(pg)

            hess = hht[t][t]
            if hess != 0.0:
                w[i][t] = min(w_orig[i][t], max(w[i][t] - grad / hess, 0.0))

    return violation
