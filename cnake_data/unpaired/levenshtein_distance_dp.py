"""Levenshtein distance kernels for single and pairwise token sequences."""

from __future__ import annotations


def calc_min_two(a: int, b: int) -> int:
    return a if a <= b else b


def calc_min_three(a: int, b: int, c: int) -> int:
    m = calc_min_two(a, b)
    return c if c < m else m


def levenshtein_dist_single(ds: list[int], start1: int, size1: int, start2: int, size2: int) -> int:
    if size1 == 0:
        return size2
    if size2 == 0:
        return size1

    distances = [0] * (size1 + 1)
    distances[0] = 0

    for i in range(1, size1 + 1):
        distances[i] = calc_min_two(distances[i - 1], i - 1)
        if ds[start1 + i - 1] != ds[start2]:
            distances[i] += 1

    dist = 0
    for j in range(1, size2):
        dist = j + 1
        for i in range(1, size1 + 1):
            temp = calc_min_three(dist, distances[i - 1], distances[i])
            if ds[start1 + i - 1] != ds[start2 + j]:
                temp += 1
            distances[i - 1] = dist
            dist = temp

    return dist


def levenshtein_dist_pairwise(doc_indices: list[int], ds: list[int]) -> list[tuple[int, int, int]]:
    out: list[tuple[int, int, int]] = []
    n_docs = len(doc_indices) - 1
    for i in range(n_docs):
        s1 = doc_indices[i]
        n1 = doc_indices[i + 1] - doc_indices[i]
        for j in range(i + 1, n_docs):
            s2 = doc_indices[j]
            n2 = doc_indices[j + 1] - doc_indices[j]
            d = levenshtein_dist_single(ds, s1, n1, s2, n2)
            out.append((i, j, d))
    return out
