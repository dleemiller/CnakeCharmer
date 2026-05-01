"""Sparse bandwidth/profile and CSR/CSC permutation helpers."""

from __future__ import annotations


def sparse_bandwidth(idx, ptr, nrows):
    lb = -nrows
    ub = -nrows
    mb = 0
    for ii in range(nrows):
        for jj in range(ptr[ii], ptr[ii + 1]):
            ldist = ii - idx[jj]
            lb = max(lb, ldist)
            ub = max(ub, -ldist)
            mb = max(mb, ub + lb + 1)
    return mb, lb, ub


def sparse_profile(idx, ptr, nrows):
    pro = 0
    for ii in range(nrows):
        temp = 0
        for jj in range(ptr[ii], ptr[ii + 1]):
            ldist = idx[jj] - ii
            temp = max(temp, ldist)
        pro += temp
    return pro
