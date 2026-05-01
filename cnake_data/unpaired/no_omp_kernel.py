"""Nested-loop CPU kernel placeholder."""

from __future__ import annotations


def ffdv_new(np1):
    acc = 0
    for n1 in range(1, np1):
        for n2 in range(1, np1):
            acc += (n1 * n2 * n1 * n2) // 100 + 341
    return acc


def doit(rep, tr, n):
    out = 0
    for _ in range(rep):
        for _ in range(tr):
            out += ffdv_new(n)
    return out
