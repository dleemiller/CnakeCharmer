from __future__ import annotations

import math

NEG_INF = float("-inf")


def to_log_prob(value: float) -> float:
    return NEG_INF if value == 0.0 else math.log(value)


def logprob_add(x: float, y: float) -> float:
    if x == NEG_INF:
        return y
    if y == NEG_INF:
        return x
    if x < y:
        x, y = y, x
    return x + math.log1p(math.exp(y - x))


def logprob_mul(x: float, y: float) -> float:
    return x + y


def logprob_div(x: float, y: float) -> float:
    return x - y
