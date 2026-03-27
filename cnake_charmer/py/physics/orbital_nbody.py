"""Simulate 3-body orbital system using Verlet integration.

Keywords: physics, orbital, nbody, gravity, simulation, verlet, benchmark
"""

import math

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(200000,))
def orbital_nbody(n: int) -> tuple:
    """Simulate a 3-body gravitational system for n timesteps.

    Three bodies with masses 1.0, 0.5, 0.3 at initial positions forming
    a triangle. Uses velocity Verlet integration with dt=1e-4 and G=1.0.
    Returns (final_x1, final_y1, total_energy) at the end.

    Args:
        n: Number of timesteps.

    Returns:
        Tuple of (final_x1, final_y1, total_energy).
    """
    G = 1.0
    dt = 1e-4
    nb = 3
    softening = 1e-6

    # masses
    mass = [1.0, 0.5, 0.3]

    # positions (x, y)
    px = [0.0, 1.0, -0.5]
    py_ = [0.0, 0.0, 0.866]

    # velocities
    vx = [0.0, 0.0, 0.0]
    vy = [0.1, -0.2, 0.1]

    # accelerations
    ax = [0.0, 0.0, 0.0]
    ay = [0.0, 0.0, 0.0]

    # Compute initial accelerations
    for i in range(nb):
        ax[i] = 0.0
        ay[i] = 0.0
        for j in range(nb):
            if i == j:
                continue
            dx = px[j] - px[i]
            dy = py_[j] - py_[i]
            r2 = dx * dx + dy * dy + softening
            r = math.sqrt(r2)
            r3 = r2 * r
            f = G * mass[j] / r3
            ax[i] += f * dx
            ay[i] += f * dy

    for _ in range(n):
        # Update positions
        for i in range(nb):
            px[i] = px[i] + vx[i] * dt + 0.5 * ax[i] * dt * dt
            py_[i] = py_[i] + vy[i] * dt + 0.5 * ay[i] * dt * dt

        # Compute new accelerations
        ax_new = [0.0, 0.0, 0.0]
        ay_new = [0.0, 0.0, 0.0]
        for i in range(nb):
            for j in range(nb):
                if i == j:
                    continue
                dx = px[j] - px[i]
                dy = py_[j] - py_[i]
                r2 = dx * dx + dy * dy + softening
                r = math.sqrt(r2)
                r3 = r2 * r
                f = G * mass[j] / r3
                ax_new[i] += f * dx
                ay_new[i] += f * dy

        # Update velocities
        for i in range(nb):
            vx[i] = vx[i] + 0.5 * (ax[i] + ax_new[i]) * dt
            vy[i] = vy[i] + 0.5 * (ay[i] + ay_new[i]) * dt
            ax[i] = ax_new[i]
            ay[i] = ay_new[i]

    # Compute total energy
    ke = 0.0
    for i in range(nb):
        ke += 0.5 * mass[i] * (vx[i] * vx[i] + vy[i] * vy[i])

    pe = 0.0
    for i in range(nb):
        for j in range(i + 1, nb):
            dx = px[j] - px[i]
            dy = py_[j] - py_[i]
            r = math.sqrt(dx * dx + dy * dy + softening)
            pe -= G * mass[i] * mass[j] / r

    total_energy = ke + pe
    return (px[0], py_[0], total_energy)
