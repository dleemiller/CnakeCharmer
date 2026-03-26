"""Deterministic quasi-Monte Carlo integration using Halton sequences.

Keywords: simulation, Monte Carlo, quasi-random, Halton sequence, integration, benchmark
"""

import math

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(300000,))
def monte_carlo_integration(n: int) -> tuple:
    """Estimate multi-dimensional integrals using Halton quasi-random sequences.

    Uses Halton sequences with bases 2, 3, 5 to generate deterministic
    quasi-random points in [0,1]^3. Estimates three integrals:
    1) Volume of unit sphere octant: x^2 + y^2 + z^2 <= 1
    2) Integral of sin(pi*x)*cos(pi*y)*exp(-z) over [0,1]^3
    3) Integral of sqrt(x*y*z) over [0,1]^3

    Args:
        n: Number of sample points.

    Returns:
        Tuple of (sphere_estimate, sincos_estimate, sqrt_estimate).
    """
    # Halton sequence generator (inline for pure Python)
    sphere_count = 0
    sincos_sum = 0.0
    sqrt_sum = 0.0

    pi = math.pi

    for i in range(1, n + 1):
        # Halton base 2
        x = 0.0
        f = 0.5
        idx = i
        while idx > 0:
            x += f * (idx % 2)
            idx //= 2
            f *= 0.5

        # Halton base 3
        y = 0.0
        f = 1.0 / 3.0
        idx = i
        while idx > 0:
            y += f * (idx % 3)
            idx //= 3
            f /= 3.0

        # Halton base 5
        z = 0.0
        f = 0.2
        idx = i
        while idx > 0:
            z += f * (idx % 5)
            idx //= 5
            f *= 0.2

        # Sphere octant test
        r2 = x * x + y * y + z * z
        if r2 <= 1.0:
            sphere_count += 1

        # sin*cos*exp integral
        sincos_sum += math.sin(pi * x) * math.cos(pi * y) * math.exp(-z)

        # sqrt integral
        sqrt_sum += math.sqrt(x * y * z + 1e-30)

    sphere_estimate = sphere_count / n
    sincos_estimate = sincos_sum / n
    sqrt_estimate = sqrt_sum / n

    return (sphere_estimate, sincos_estimate, sqrt_estimate)
