"""N-body gravity simulation with GIL release.

Keywords: simulation, n-body, gravity, nogil, physics, benchmark
"""

import math

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(500,))
def nogil_nbody_step(n: int) -> float:
    """Simulate n particles under gravity for 10 steps.

    Initial positions: x=sin(i*0.1), y=cos(i*0.1).
    All velocities start at zero. G=1.0, softening=0.01,
    dt=0.0005. Returns total kinetic energy after 10 steps.

    Args:
        n: Number of particles.

    Returns:
        Total kinetic energy as a float.
    """
    dt = 0.0005
    soft2 = 0.0001  # softening^2
    steps = 10

    px = [math.sin(i * 0.1) for i in range(n)]
    py = [math.cos(i * 0.1) for i in range(n)]
    vx = [0.0] * n
    vy = [0.0] * n

    for _step in range(steps):
        ax = [0.0] * n
        ay = [0.0] * n

        for i in range(n):
            for j in range(i + 1, n):
                dx = px[j] - px[i]
                dy = py[j] - py[i]
                dist2 = dx * dx + dy * dy + soft2
                inv_dist3 = 1.0 / (dist2 * math.sqrt(dist2))
                fx = dx * inv_dist3
                fy = dy * inv_dist3
                ax[i] += fx
                ay[i] += fy
                ax[j] -= fx
                ay[j] -= fy

        for i in range(n):
            vx[i] += ax[i] * dt
            vy[i] += ay[i] * dt
            px[i] += vx[i] * dt
            py[i] += vy[i] * dt

    ke = 0.0
    for i in range(n):
        ke += 0.5 * (vx[i] * vx[i] + vy[i] * vy[i])

    return ke
