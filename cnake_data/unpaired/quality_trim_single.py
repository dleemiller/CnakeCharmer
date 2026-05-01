"""Single-end quality trim index using BWA-like criterion."""

from __future__ import annotations


def quality_trim_index(qualities, cutoff, base=33):
    s = 0
    max_qual = 0
    max_i = len(qualities)
    for i in range(len(qualities) - 1, -1, -1):
        s += cutoff - (ord(qualities[i]) - base)
        if s < 0:
            break
        if s > max_qual:
            max_qual = s
            max_i = i
    return max_i
