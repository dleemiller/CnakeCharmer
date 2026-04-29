"""Particle evolution under tangential velocity field."""

from __future__ import annotations

import math


def c_evolve(r_i, ang_speed_i, timestep, nsteps):
    nparticles = r_i.shape[0]
    for _ in range(nsteps):
        for j in range(nparticles):
            x = r_i[j, 0]
            y = r_i[j, 1]
            ang_speed = ang_speed_i[j]
            norm = math.sqrt(x**2 + y**2)
            vx = (-y) / norm
            vy = x / norm
            dx = timestep * ang_speed * vx
            dy = timestep * ang_speed * vy
            r_i[j, 0] += dx
            r_i[j, 1] += dy


def c_evolve_openmp(r_i, ang_speed_i, timestep, nsteps):
    # Serial Python equivalent of per-particle independent evolution.
    nparticles = r_i.shape[0]
    for j in range(nparticles):
        for _ in range(nsteps):
            x = r_i[j, 0]
            y = r_i[j, 1]
            ang_speed = ang_speed_i[j]
            norm = math.sqrt(x**2 + y**2)
            vx = (-y) / norm
            vy = x / norm
            dx = timestep * ang_speed * vx
            dy = timestep * ang_speed * vy
            r_i[j, 0] += dx
            r_i[j, 1] += dy
