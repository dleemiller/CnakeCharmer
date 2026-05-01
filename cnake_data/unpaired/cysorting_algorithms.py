"""Insertion sort and merge sort (ascending/descending)."""

from __future__ import annotations


def insertion_sort(x: list, reverse: bool = False):
    n = len(x)
    for i in range(n):
        elem = x[i]
        j = i - 1
        if reverse:
            while j >= 0 and x[j] < elem:
                x[j + 1] = x[j]
                j -= 1
        else:
            while j >= 0 and x[j] > elem:
                x[j + 1] = x[j]
                j -= 1
        x[j + 1] = elem
    return x


def merge_asc(left: list, right: list):
    sl, sr = len(left), len(right)
    n = sl + sr
    merged = [None] * n
    i = j = 0
    for k in range(n):
        if i < sl and (j >= sr or left[i] < right[j]):
            merged[k] = left[i]
            i += 1
        else:
            merged[k] = right[j]
            j += 1
    return merged


def merge_desc(left: list, right: list):
    sl, sr = len(left), len(right)
    n = sl + sr
    merged = [None] * n
    i = j = 0
    for k in range(n):
        if i < sl and (j >= sr or left[i] > right[j]):
            merged[k] = left[i]
            i += 1
        else:
            merged[k] = right[j]
            j += 1
    return merged


def merge_sort(x: list, reverse: bool = False):
    n = len(x)
    if n <= 75:
        return insertion_sort(x, reverse=reverse)
    merge = merge_desc if reverse else merge_asc
    mid = n // 2
    left = merge_sort(x[:mid], reverse=reverse)
    right = merge_sort(x[mid:], reverse=reverse)
    return merge(left, right)
