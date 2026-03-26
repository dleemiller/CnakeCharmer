"""N-body simulation with Verlet integration.

Keywords: n-body, Verlet integration, physics simulation, benchmark
"""

import math

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(500,))
def nbody_verlet(n: int) -> float:
    """N-body simulation with Verlet integration for 100 timesteps.

    Initial positions: x[i]=sin(i), y[i]=cos(i). Mass=1, dt=0.001.
    Gravitational constant G=1.0, softening=0.1.

    Args:
        n: Number of bodies.

    Returns:
        Total kinetic energy after simulation as a float.
    """
    dt = 0.001
    softening = 0.1
    soft2 = softening * softening
    steps = 100

    # Initialize positions and velocities
    px = [math.sin(i) for i in range(n)]
    py_arr = [math.cos(i) for i in range(n)]
    vx = [0.0] * n
    vy = [0.0] * n

    # Compute initial accelerations
    ax = [0.0] * n
    ay = [0.0] * n
    for i in range(n):
        for j in range(i + 1, n):
            dx = px[j] - px[i]
            dy = py_arr[j] - py_arr[i]
            dist2 = dx * dx + dy * dy + soft2
            inv_dist3 = 1.0 / (dist2 * math.sqrt(dist2))
            fx = dx * inv_dist3
            fy = dy * inv_dist3
            ax[i] += fx
            ay[i] += fy
            ax[j] -= fx
            ay[j] -= fy

    # Verlet integration loop
    for _step in range(steps):
        # Update positions
        for i in range(n):
            px[i] += vx[i] * dt + 0.5 * ax[i] * dt * dt
            py_arr[i] += vy[i] * dt + 0.5 * ay[i] * dt * dt

        # Compute new accelerations
        ax_new = [0.0] * n
        ay_new = [0.0] * n
        for i in range(n):
            for j in range(i + 1, n):
                dx = px[j] - px[i]
                dy = py_arr[j] - py_arr[i]
                dist2 = dx * dx + dy * dy + soft2
                inv_dist3 = 1.0 / (dist2 * math.sqrt(dist2))
                fx = dx * inv_dist3
                fy = dy * inv_dist3
                ax_new[i] += fx
                ay_new[i] += fy
                ax_new[j] -= fx
                ay_new[j] -= fy

        # Update velocities
        for i in range(n):
            vx[i] += 0.5 * (ax[i] + ax_new[i]) * dt
            vy[i] += 0.5 * (ay[i] + ay_new[i]) * dt

        ax = ax_new
        ay = ay_new

    # Total kinetic energy
    ke = 0.0
    for i in range(n):
        ke += 0.5 * (vx[i] * vx[i] + vy[i] * vy[i])

    return ke
