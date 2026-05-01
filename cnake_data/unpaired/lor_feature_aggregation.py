from __future__ import annotations


def aggregate_sparse_features(
    data: list[list[tuple[float, float]]],
    spar_ind: list[list[int]],
    ns: list[float],
    ks: list[float],
) -> tuple[list[list[float]], list[float]]:
    num_samp = len(data)
    feat = [[0.0 for _ in range(400)] for _ in range(num_samp)]
    for i in range(num_samp):
        for j in range(min(100, len(spar_ind[i]))):
            idx = spar_ind[i][j]
            if idx == -1:
                break
            a, b = data[i][j]
            base = 4 * idx
            feat[i][base] = a
            feat[i][base + 1] = b
            feat[i][base + 2] = ns[idx]
            feat[i][base + 3] = ks[idx]
    kappa_eta = [0.0] * 100
    for j in range(min(100, len(ns), len(ks))):
        if ns[j] != 0.0:
            kappa_eta[j] = ks[j] / ns[j]
    return feat, kappa_eta
