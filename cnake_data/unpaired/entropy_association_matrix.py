from __future__ import annotations

import math
import random


def entropy_binary(vec: list[int]) -> float:
    n = len(vec)
    if n == 0:
        return 0.0
    p1 = sum(1 for v in vec if v) / float(n)
    p0 = 1.0 - p1

    def plog(p: float) -> float:
        return 0.0 if p == 0.0 else p * math.log(p)

    return -(plog(p0) + plog(p1))


def and_entropy(a: list[int], b: list[int]) -> float:
    c = [1 if (a[i] and b[i]) else 0 for i in range(min(len(a), len(b)))]
    return entropy_binary(c)


def z_score_entropy(a: list[int], b: list[int], times: int = 10) -> float:
    x = and_entropy(a, b)
    vals: list[float] = []
    b2 = b[:]
    for _ in range(times):
        random.shuffle(b2)
        vals.append(and_entropy(a, b2))
    mean = sum(vals) / float(len(vals)) if vals else 0.0
    var = sum((v - mean) * (v - mean) for v in vals) / float(len(vals)) if vals else 0.0
    std = math.sqrt(var)
    if std == 0.0:
        return 0.0
    return (x - mean) / std
