"""Frequency-weight normalization helpers."""

from __future__ import annotations


def _calc_maxweight(keyweights):
    feedback = 0.0
    for _, weight in keyweights:
        if weight > feedback:
            feedback = weight
    return feedback


def make_weights_correct(keyweights):
    maxweight = _calc_maxweight(keyweights)
    out = []
    for freq, weight in keyweights:
        try:
            out.append((freq, weight / maxweight))
        except ZeroDivisionError:
            out.append((freq, 1.0))
    return out
