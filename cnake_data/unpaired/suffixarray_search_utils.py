from __future__ import annotations


def suffix_cmp(p: tuple[int, ...], t: list[int], sa: list[int], n: int, c: int) -> int:
    i = 0
    while i < len(p) and (sa[c] + i) < n:
        a = p[i]
        b = t[sa[c] + i]
        if a > b:
            return 1
        if a < b:
            return -1
        i += 1
    return 0


def suffix_range(p: tuple[int, ...], t: list[int], sa: list[int], n: int) -> tuple[int, int]:
    left = 0
    r = n
    while left < r:
        c = (left + r) // 2
        if suffix_cmp(p, t, sa, n, c) > 0:
            left = c + 1
        else:
            r = c
    s = left
    r = n
    while left < r:
        c = (left + r) // 2
        if suffix_cmp(p, t, sa, n, c) < 0:
            r = c
        else:
            left = c + 1
    return s, r
