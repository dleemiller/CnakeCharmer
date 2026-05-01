from __future__ import annotations

import math


def sigmoid(z: float) -> float:
    if z > 6.0:
        return 1.0
    if z < -6.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-z))


def update_user_item(
    u_vec: list[float],
    v_vec: list[float],
    val: float,
    lambda_reg: float,
    lr: float,
) -> tuple[list[float], list[float], float]:
    s = 0.0
    k = len(u_vec)
    for i in range(k):
        s += u_vec[i] * v_vec[i]
    sg = sigmoid(s)
    err = val - sg
    werr = err * sg * (1.0 - sg)

    new_u = u_vec[:]
    new_v = v_vec[:]
    for i in range(k):
        gu = werr * v_vec[i] - lambda_reg * u_vec[i]
        gv = werr * u_vec[i] - lambda_reg * v_vec[i]
        new_u[i] += lr * gu
        new_v[i] += lr * gv
    return new_u, new_v, err * err
