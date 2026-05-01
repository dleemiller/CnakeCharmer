"""Contiguous sequence sum kernel."""

from __future__ import annotations


def contiguous_sum(seq):
    total = 0.0
    for value in seq:
        total += float(value)
    return total
