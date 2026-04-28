"""2D compressible Euler flux and Jacobian-eigenvalue helpers."""

from __future__ import annotations

import numpy as np

GAMMA = 1.4


def _as_u(U) -> np.ndarray:
    u = np.asarray(U, dtype=np.float64)
    if u.shape != (4,):
        raise ValueError("U must have shape (4,)")
    return u


def eigval_max_min_f(U, minim: bool = False) -> float:
    u = _as_u(U)
    u1, u2, u3, u4 = u
    c_ = 1.0 - GAMMA
    jac = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [
                (-(u2**2) / u1**2 + c_ / 2.0 * (u2**2 / u1**2 + u3**2 / u1**2)),
                (2.0 * u2 / u1 - c_ * u2 / u1),
                (-c_ * u3 / u1),
                c_,
            ],
            [(-u2 * u3 / u1**2), u3 / u1, u2 / u1, 0.0],
            [
                (-GAMMA * u4 * u2 / u1**2 + c_ * (u2**3 / u1**3 + u3**3 / u1**3)),
                (GAMMA * u4 / u1 - 1.5 * c_ * u2**2 / u1**2),
                (-1.5 * c_ * u3**2 / u1**2),
                GAMMA * u2 / u1,
            ],
        ],
        dtype=np.float64,
    )
    vals = np.real(np.linalg.eigvals(jac))
    return float(vals.min() if minim else vals.max())


def eigval_max_min_g(U, minim: bool = False) -> float:
    u = _as_u(U)
    u1, u2, u3, u4 = u
    c_ = 1.0 - GAMMA
    jac = np.array(
        [
            [0.0, 0.0, 1.0, 0.0],
            [(-u2 * u3 / u1**2), u3 / u1, u2 / u1, 0.0],
            [
                (-(u3**2) / u1**2 + c_ / 2.0 * (u2**2 / u1**2 + u3**2 / u1**2)),
                -c_ * u3 / u1,
                (2.0 * u3 / u1 - c_ * u3 / u1),
                c_,
            ],
            [
                (-GAMMA * u4 * u3 / u1**2 + c_ * (u2**3 / u1**3 + u3**3 / u1**3)),
                (-1.5 * c_ * u2**2 / u1**2),
                (GAMMA * u4 / u1 - 1.5 * c_ * u3**2 / u1**2),
                GAMMA * u3 / u1,
            ],
        ],
        dtype=np.float64,
    )
    vals = np.real(np.linalg.eigvals(jac))
    return float(vals.min() if minim else vals.max())


def flux_f(U) -> np.ndarray:
    u = _as_u(U)
    flux = np.zeros(4, dtype=np.float64)
    flux[0] = u[1]
    flux[1] = u[1] ** 2 / u[0] + (GAMMA - 1.0) * (
        u[3] - 0.5 * u[1] ** 2 / u[0] - 0.5 * u[2] ** 2 / u[0]
    )
    flux[2] = u[1] * u[2] / u[0]
    flux[3] = (
        u[1]
        / u[0]
        * (GAMMA * u[3] - (GAMMA - 1.0) * (0.5 * u[1] ** 2 / u[0] - 0.5 * u[2] ** 2 / u[0]))
    )
    return flux


def flux_g(U) -> np.ndarray:
    u = _as_u(U)
    flux = np.zeros(4, dtype=np.float64)
    flux[0] = u[2]
    flux[1] = u[1] * u[2] / u[0]
    flux[2] = u[2] ** 2 / u[0] + (GAMMA - 1.0) * (
        u[3] - 0.5 * u[1] ** 2 / u[0] - 0.5 * u[2] ** 2 / u[0]
    )
    flux[3] = (
        u[2]
        / u[0]
        * (GAMMA * u[3] - (GAMMA - 1.0) * (0.5 * u[1] ** 2 / u[0] - 0.5 * u[2] ** 2 / u[0]))
    )
    return flux
