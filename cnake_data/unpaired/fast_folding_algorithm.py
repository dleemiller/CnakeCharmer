"""Core Fast Folding Algorithm (FFA) shift-add stages."""

from __future__ import annotations

import math


def ffa_group_shift_add(group0, n_row_group, n_col_group):
    group = [[0.0 for _ in range(n_col_group)] for _ in range(n_row_group)]
    n_half = n_row_group // 2
    for i in range(n_row_group):
        i_a = i // 2
        i_b = i_a + n_half
        b_s = (i + 1) // 2
        for j in range(n_col_group):
            j_b = (j + b_s + n_col_group) % n_col_group
            group[i][j] = group0[i_a][j] + group0[i_b][j_b]
    return group


def ffa_shift_add(xw0, stage):
    n_row = len(xw0)
    n_col = len(xw0[0])
    n_row_group = 2**stage
    n_group = n_row // n_row_group
    xw = [[0.0 for _ in range(n_col)] for _ in range(n_row)]
    for ig in range(n_group):
        start = ig * n_row_group
        stop = (ig + 1) * n_row_group
        grp = ffa_group_shift_add(xw0[start:stop], n_row_group, n_col)
        xw[start:stop] = grp
    return xw


def ffa(xw):
    n_row = len(xw)
    n_stage = int(round(math.log(n_row, 2)))
    xwfs = [row[:] for row in xw]
    for stage in range(1, n_stage + 1):
        xwfs = ffa_shift_add(xwfs, stage)
    return xwfs
