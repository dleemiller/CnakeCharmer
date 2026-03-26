"""Velocity Verlet integration for harmonic oscillator.

Keywords: ODE, Verlet, integration, harmonic oscillator, differential equation, numerical
"""

import math

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(10000000,))
def verlet_integration(n: int) -> tuple:
    """Integrate harmonic oscillator x''=-x using velocity Verlet for n steps.

    Initial conditions: x(0)=1, v(0)=0.
    Integration from t=0 to t=20*pi.
    Returns (final_x, final_v, total_energy_drift) where energy = 0.5*(v^2 + x^2).

    Args:
        n: Number of integration steps.

    Returns:
        Tuple of (final_x, final_v, energy_drift).
    """
    dt = 20.0 * math.pi / n
    x = 1.0
    v = 0.0

    # Initial energy
    e0 = 0.5 * (v * v + x * x)
    energy_drift_sum = 0.0

    for _ in range(n):
        # Velocity Verlet: a = -x (harmonic oscillator)
        a = -x
        x = x + v * dt + 0.5 * a * dt * dt
        a_new = -x
        v = v + 0.5 * (a + a_new) * dt

        # Accumulate energy drift
        e = 0.5 * (v * v + x * x)
        energy_drift_sum += (e - e0) if (e - e0) >= 0 else -(e - e0)

    return (x, v, energy_drift_sum)
