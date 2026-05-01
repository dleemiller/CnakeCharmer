"""N-body simulation stepper using velocity Verlet integration."""

from __future__ import annotations

import math


def _compute_acc(n, G, m, x, y, z, offset=0):
    axm = [0.0] * n
    aym = [0.0] * n
    azm = [0.0] * n
    for j in range(n):
        ax = ay = az = 0.0
        xj, yj, zj = x[offset + j], y[offset + j], z[offset + j]
        for f in range(n):
            if f == j:
                continue
            dx = x[offset + f] - xj
            dy = y[offset + f] - yj
            dz = z[offset + f] - zj
            r = math.sqrt(dx * dx + dy * dy + dz * dz)
            invr3 = 1.0 / (r * r * r)
            ax += m[f] * G * dx * invr3
            ay += m[f] * G * dy * invr3
            az += m[f] * G * dz * invr3
        axm[j], aym[j], azm[j] = ax, ay, az
    return axm, aym, azm


def cython_solver(n, G, dt, m, M_, x, y, z, vx, vy, vz, axm, aym, azm):
    a0x, a0y, a0z = _compute_acc(n, G, m, x, y, z, 0)
    for j in range(n):
        axm[j], aym[j], azm[j] = a0x[j], a0y[j], a0z[j]

    for i in range(1, M_):
        prev = (i - 1) * n
        cur = i * n
        for j in range(n):
            x[cur + j] = x[prev + j] + vx[prev + j] * dt + 0.5 * axm[prev + j] * dt * dt
            y[cur + j] = y[prev + j] + vy[prev + j] * dt + 0.5 * aym[prev + j] * dt * dt
            z[cur + j] = z[prev + j] + vz[prev + j] * dt + 0.5 * azm[prev + j] * dt * dt

        acx, acy, acz = _compute_acc(n, G, m, x, y, z, cur)
        for j in range(n):
            axm[cur + j], aym[cur + j], azm[cur + j] = acx[j], acy[j], acz[j]
            vx[cur + j] = vx[prev + j] + 0.5 * dt * (axm[cur + j] + axm[prev + j])
            vy[cur + j] = vy[prev + j] + 0.5 * dt * (aym[cur + j] + aym[prev + j])
            vz[cur + j] = vz[prev + j] + 0.5 * dt * (azm[cur + j] + azm[prev + j])

    return x, y, z, vx, vy, vz, axm, aym, azm
