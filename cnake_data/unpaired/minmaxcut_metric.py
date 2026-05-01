"""MinMaxCut clustering metric from adjacency/cut primitives."""

from __future__ import annotations


def evaluate_minmaxcut(
    clusters,
    condensed_matrix,
    get_cluster_and_complementary,
    cut,
    calc_adjacency_matrix,
    internal_vol,
):
    W, D = calc_adjacency_matrix(condensed_matrix)
    mmcut_val = 0.0
    for i in range(len(clusters)):
        A, Acomp = get_cluster_and_complementary(i, clusters)
        mmcut_val += cut(A, Acomp, W) / internal_vol(A, D, condensed_matrix)
    return 0.5 * mmcut_val
