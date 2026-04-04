"""Compute total gravitational potential energy of n bodies.

Keywords: numerical, n-body, gravitational, physics, pairwise, benchmark
"""

import math

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(2000,))
def nbody_energy(n: int) -> float:
    """Compute total gravitational potential energy of n bodies.

    Positions: x[i] = sin(i*0.1), y[i] = cos(i*0.2), z[i] = sin(i*0.3).
    Mass: mass[i] = 1.0 + (i%5)*0.5.
    Returns sum of all pairwise -G*m1*m2/r where G=6.674e-11.

    Args:
        n: Number of bodies.

    Returns:
        Total gravitational potential energy as a float.
    """
    G = 6.674e-11
    xs = [math.sin(i * 0.1) for i in range(n)]
    ys = [math.cos(i * 0.2) for i in range(n)]
    zs = [math.sin(i * 0.3) for i in range(n)]
    masses = [1.0 + (i % 5) * 0.5 for i in range(n)]

    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dz = zs[i] - zs[j]
            r = math.sqrt(dx * dx + dy * dy + dz * dz)
            if r > 0:
                total += -G * masses[i] * masses[j] / r

    return total
