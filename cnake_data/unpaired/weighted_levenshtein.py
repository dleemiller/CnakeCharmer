"""Weighted Levenshtein distance between numeric sequences."""

from __future__ import annotations


def uniform_matrix(r, c, v=0):
    return [[v for _ in range(c)] for _ in range(r)]


def lev_rec(seq1, seq2, i: int, j: int, memo):
    if min(i, j) == 0:
        if i > j:
            return sum(seq1[0:i]) + i
        return sum(seq2[0:j]) + j
    if memo[i - 1][j - 1] != -1:
        return memo[i - 1][j - 1]

    lev_del = lev_rec(seq1, seq2, i - 1, j, memo) + 1 + seq1[i - 1]
    lev_ins = lev_rec(seq1, seq2, i, j - 1, memo) + 1 + seq2[j - 1]
    lev_sub = lev_rec(seq1, seq2, i - 1, j - 1, memo) + abs(seq1[i - 1] - seq2[j - 1])
    dist = min(lev_del, lev_ins, lev_sub)
    memo[i - 1][j - 1] = dist
    return dist


def lev_distance(pair):
    seq1, seq2 = pair
    i_init = len(seq1)
    j_init = len(seq2)
    memo = uniform_matrix(i_init, j_init, -1)
    return lev_rec(seq1, seq2, i_init, j_init, memo)
