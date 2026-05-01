"""Grouped sum and mean helpers using integer index windows."""

from __future__ import annotations


def grouped_sum(data, indexes, start, end):
    total = 0
    for i in range(start, end):
        total += data[indexes[i]]
    return total


def grouped_mean(data, indexes, start, end):
    size = end - start
    if size <= 0:
        raise ValueError("empty window")
    return grouped_sum(data, indexes, start, end) / size
