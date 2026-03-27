"""Simulate a spring-damper (mass-spring-dashpot) system.

Keywords: physics, spring, damper, dashpot, oscillation, simulation, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(500000,))
def spring_damper(n: int) -> tuple:
    """Simulate a spring-damper system for n timesteps using Euler integration.

    Parameters: mass=1.0, spring constant k=50.0, damping c=0.5,
    initial displacement x0=2.0, initial velocity v0=0.0, dt=1e-5.
    Tracks maximum displacement magnitude over the simulation.

    Args:
        n: Number of timesteps.

    Returns:
        Tuple of (final_x, final_v, max_displacement).
    """
    m = 1.0
    k = 50.0
    c = 0.5
    dt = 1e-5

    x = 2.0
    v = 0.0
    max_disp = 2.0

    for _ in range(n):
        a = (-k * x - c * v) / m
        v = v + a * dt
        x = x + v * dt
        ax = x if x >= 0.0 else -x
        if ax > max_disp:
            max_disp = ax

    return (x, v, max_disp)
