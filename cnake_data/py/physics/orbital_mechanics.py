"""Compute orbital periods for planets using Kepler's third law.

Keywords: physics, orbital, kepler, period, gravitational, benchmark
"""

import math

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(5000000,))
def orbital_mechanics(n: int) -> float:
    """Compute orbital periods for n planets using Kepler's third law.

    Semi-major axis a[i] = 0.5 + i*0.3 AU, central mass = 1 solar mass.
    T = 2*pi*sqrt(a^3 / GM). Returns sum of all periods.

    Args:
        n: Number of planets.

    Returns:
        Sum of orbital periods as a float.
    """
    G = 6.674e-11
    M_sun = 1.989e30
    AU = 1.496e11
    GM = G * M_sun
    two_pi = 2.0 * math.pi

    total = 0.0
    for i in range(n):
        a = (0.5 + i * 0.3) * AU
        T = two_pi * math.sqrt(a * a * a / GM)
        total += T

    return total
