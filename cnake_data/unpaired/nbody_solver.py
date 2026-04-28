"""Simple N-body timestep integrator."""

from __future__ import annotations

from math import sqrt


def solver(n, G, dt, m, t, x, y, vx, vy, axm, aym):
    for j in range(n):
        ax = 0.0
        ay = 0.0
        for f in range(n):
            if f != j:
                dx = x[f] - x[j]
                dy = y[f] - y[j]
                r3 = sqrt(dx * dx + dy * dy) ** 3
                ax += m[f] * G * dx / r3
                ay += m[f] * G * dy / r3
        axm[j] = ax
        aym[j] = ay

    for i in range(len(t)):
        if i != 0:
            for j in range(n):
                x[i * n + j] = (
                    x[(i - 1) * n + j]
                    + vx[(i - 1) * n + j] * dt
                    + 0.5 * axm[(i - 1) * n + j] * dt * dt
                )
                y[i * n + j] = (
                    y[(i - 1) * n + j]
                    + vy[(i - 1) * n + j] * dt
                    + 0.5 * aym[(i - 1) * n + j] * dt * dt
                )

            for j in range(n):
                ax = 0.0
                ay = 0.0
                for f in range(n):
                    if f != j:
                        dx = x[i * n + f] - x[i * n + j]
                        dy = y[i * n + f] - y[i * n + j]
                        r3 = sqrt(dx * dx + dy * dy) ** 3
                        ax += m[f] * G * dx / r3
                        ay += m[f] * G * dy / r3
                axm[i * n + j] = ax
                aym[i * n + j] = ay
                vx[i * n + j] = vx[(i - 1) * n + j] + 0.5 * dt * (
                    axm[i * n + j] + axm[(i - 1) * n + j]
                )
                vy[i * n + j] = vy[(i - 1) * n + j] + 0.5 * dt * (
                    aym[i * n + j] + aym[(i - 1) * n + j]
                )

    return x, y, vx, vy, axm, aym
