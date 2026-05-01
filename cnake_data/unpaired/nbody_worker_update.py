"""N-body update kernel computing accelerations, positions, and velocities."""

from __future__ import annotations

import math

G = 6.67408e-11


def update_planet_indices(
    o_positions: list[list[float]],
    o_speeds: list[list[float]],
    o_accels: list[list[float]],
    o_masses: list[float],
    num_planets: int,
    i_from: int,
    i_to: int,
    delta_t: float,
) -> tuple[list[list[float]], list[list[float]], list[list[float]]]:
    span = i_to - i_from
    delta_t_sq_half = (delta_t * delta_t) / 2.0

    r_pos = [[0.0, 0.0, 0.0] for _ in range(span)]
    r_speeds = [[0.0, 0.0, 0.0] for _ in range(span)]
    r_accels = [[0.0, 0.0, 0.0] for _ in range(span)]

    for off in range(span):
        i = i_from + off
        resulting_force = [0.0, 0.0, 0.0]

        for j in range(num_planets):
            if i == j:
                continue
            v0 = o_positions[j][0] - o_positions[i][0]
            v1 = o_positions[j][1] - o_positions[i][1]
            v2 = o_positions[j][2] - o_positions[i][2]
            dist = math.sqrt(v0 * v0 + v1 * v1 + v2 * v2)
            if dist == 0.0:
                continue

            tmp = G * ((o_masses[i] * o_masses[j]) / (dist**3))
            resulting_force[0] += tmp * v0
            resulting_force[1] += tmp * v1
            resulting_force[2] += tmp * v2

        for dim in range(3):
            accel = resulting_force[dim] / o_masses[i]
            r_accels[off][dim] = accel
            r_pos[off][dim] = (
                o_positions[i][dim]
                + delta_t * o_speeds[i][dim]
                + delta_t_sq_half * o_accels[i][dim]
            )
            r_speeds[off][dim] = o_speeds[i][dim] + accel * delta_t

    return r_pos, r_speeds, r_accels


def move_planets(
    o_positions: list[list[float]],
    o_speeds: list[list[float]],
    o_accels: list[list[float]],
    o_masses: list[float],
    delta_t: float,
) -> tuple[list[list[float]], list[list[float]], list[list[float]]]:
    n = len(o_positions)
    return update_planet_indices(o_positions, o_speeds, o_accels, o_masses, n, 0, n, delta_t)
