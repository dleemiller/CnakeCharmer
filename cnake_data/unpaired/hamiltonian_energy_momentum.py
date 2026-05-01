from __future__ import annotations


def total_energy(
    f: list[list[float]],
    h0: list[list[float]],
    h1: list[list[float]],
    h2: list[list[float]],
    hx: float,
    hv: float,
) -> tuple[float, float, float]:
    nx = len(f)
    nv = len(f[0]) if nx else 0
    ek = 0.0
    ep = 0.0
    for ix in range(nx):
        for iv in range(nv):
            fv = f[ix][iv]
            ek += fv * h0[ix][iv]
            ep += fv * h1[ix][iv]
            ep += fv * h2[ix][iv]
    e_kin = ek * hx * hv
    e_pot = ep * hx * hv
    e_tot = e_kin + 0.5 * e_pot
    return e_kin, e_pot, e_tot


def total_momentum(
    f: list[list[float]], v: list[float], mass: float, hx: float, hv: float
) -> float:
    nx = len(f)
    nv = len(f[0]) if nx else 0
    p = 0.0
    for ix in range(nx):
        for iv in range(max(0, nv - 1)):
            p += f[ix][iv] * v[iv]
    return p * mass * hx * hv
