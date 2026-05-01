"""Object-groupby nth/last aggregation helpers."""

from __future__ import annotations


def group_nth_object(values, labels, rank, ngroups):
    k = len(values[0]) if values else 0
    counts = [0] * ngroups
    nobs = [[0] * k for _ in range(ngroups)]
    out = [[None] * k for _ in range(ngroups)]
    for i, row in enumerate(values):
        lab = labels[i]
        if lab < 0:
            continue
        counts[lab] += 1
        for j, val in enumerate(row):
            if val == val:
                nobs[lab][j] += 1
                if nobs[lab][j] == rank:
                    out[lab][j] = val
    return out, counts


def group_last_object(values, labels, ngroups):
    k = len(values[0]) if values else 0
    counts = [0] * ngroups
    nobs = [[0] * k for _ in range(ngroups)]
    out = [[None] * k for _ in range(ngroups)]
    for i, row in enumerate(values):
        lab = labels[i]
        if lab < 0:
            continue
        counts[lab] += 1
        for j, val in enumerate(row):
            if val == val:
                nobs[lab][j] += 1
                out[lab][j] = val
    return out, counts
