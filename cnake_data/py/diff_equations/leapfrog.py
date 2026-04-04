"""Leapfrog integration for a harmonic oscillator.

Integrates x'' = -x using the leapfrog (Stormer-Verlet) method.
Returns trajectory metrics and energy drift.

Keywords: ODE, leapfrog, Verlet, harmonic oscillator, symplectic, integration, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(5000000,))
def leapfrog(n: int) -> tuple:
    """Integrate harmonic oscillator x'' = -x using leapfrog method.

    Initial conditions: x(0) = 1.0, v(0) = 0.0.
    Integration over t in [0, 50].

    Args:
        n: Number of integration steps.

    Returns:
        Tuple of (final_x, final_v, energy_drift).
    """
    dt = 50.0 / n
    x = 1.0
    v = 0.0

    # Initial energy: E = 0.5*v^2 + 0.5*x^2
    e0 = 0.5 * v * v + 0.5 * x * x

    # Half-step kick for velocity
    v = v - 0.5 * dt * x

    for _ in range(n):
        # Drift (position update)
        x = x + dt * v
        # Kick (velocity update) - force = -x
        v = v - dt * x if _ < n - 1 else v - 0.5 * dt * x

    # Final energy
    e_final = 0.5 * v * v + 0.5 * x * x
    energy_drift = abs(e_final - e0)

    return (x, v, energy_drift)
