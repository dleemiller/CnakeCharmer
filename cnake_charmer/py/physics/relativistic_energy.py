"""Compute relativistic kinetic energy for particles.

Keywords: physics, relativistic, energy, lorentz, gamma, benchmark
"""

import math

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(2000000,))
def relativistic_energy(n: int) -> float:
    """Compute relativistic kinetic energy for n particles.

    v[i] = (i*7 + 3) % 299000000 m/s (up to ~c).
    KE = (gamma - 1) * m * c^2 where gamma = 1/sqrt(1 - v^2/c^2).
    m = 1e-27 kg. Returns sum of kinetic energies.

    Args:
        n: Number of particles.

    Returns:
        Sum of kinetic energies as a float.
    """
    c = 299792458.0
    c2 = c * c
    m = 1e-27

    total = 0.0
    for i in range(n):
        v = (i * 7 + 3) % 299000000
        v2_over_c2 = (v * v) / c2
        if v2_over_c2 >= 1.0:
            continue
        gamma = 1.0 / math.sqrt(1.0 - v2_over_c2)
        KE = (gamma - 1.0) * m * c2
        total += KE

    return total
