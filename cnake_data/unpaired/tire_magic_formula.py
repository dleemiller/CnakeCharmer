from __future__ import annotations

import math


def sign(x: float) -> float:
    if x > 0.0:
        return 1.0
    if x < 0.0:
        return -1.0
    return 0.0


def longitudinal_force(kappa: float, gamma: float, fz: float, p: dict[str, float]) -> float:
    kappa = -kappa
    shx = p["p_hx1"]
    svx = fz * p["p_vx1"]
    kx = kappa + shx
    mux = p["p_dx1"] * (1.0 - p["p_dx3"] * gamma * gamma)
    cx = p["p_cx1"]
    dx = mux * fz
    ex = p["p_ex1"]
    kstiff = fz * p["p_kx1"]
    bx = kstiff / (cx * dx)
    t = bx * kx
    return dx * math.sin(cx * math.atan(t - ex * (t - math.atan(t))) + svx)


def lateral_force(
    alpha: float, gamma: float, fz: float, p: dict[str, float]
) -> tuple[float, float]:
    shy = sign(gamma) * (p["p_hy1"] + p["p_hy3"] * abs(gamma))
    svy = sign(gamma) * fz * (p["p_vy1"] + p["p_vy3"] * abs(gamma))
    ay = alpha + shy
    muy = p["p_dy1"] * (1.0 - p["p_dy3"] * gamma * gamma)
    cy = p["p_cy1"]
    dy = muy * fz
    ey = p["p_ey1"]
    ky = fz * p["p_ky1"]
    by = ky / (cy * dy)
    t = by * ay
    fy = dy * math.sin(cy * math.atan(t - ey * (t - math.atan(t)))) + svy
    return fy, muy
