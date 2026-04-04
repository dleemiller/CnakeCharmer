"""Simulate coupled harmonic oscillators using Verlet integration.

Keywords: physics, harmonic, oscillator, verlet, simulation, benchmark
"""

import math

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(3000,))
def harmonic_oscillator(n: int) -> float:
    """Simulate n coupled harmonic oscillators for 1000 steps.

    Initial: x[i] = sin(i*0.1), v[i] = 0. Spring constant k=1.0, mass m=1.0.
    Verlet integration with dt=0.001. Coupling between neighbors.
    Returns total energy (kinetic + potential) at the end.

    Args:
        n: Number of oscillators.

    Returns:
        Total energy as a float.
    """
    k = 1.0
    dt = 0.001
    steps = 1000

    x = [math.sin(i * 0.1) for i in range(n)]
    v = [0.0] * n
    a = [0.0] * n

    # Compute initial accelerations
    for i in range(n):
        force = -k * x[i]
        if i > 0:
            force += k * (x[i - 1] - x[i])
        if i < n - 1:
            force += k * (x[i + 1] - x[i])
        a[i] = force

    # Verlet integration
    for _step in range(steps):
        for i in range(n):
            x[i] += v[i] * dt + 0.5 * a[i] * dt * dt

        a_new = [0.0] * n
        for i in range(n):
            force = -k * x[i]
            if i > 0:
                force += k * (x[i - 1] - x[i])
            if i < n - 1:
                force += k * (x[i + 1] - x[i])
            a_new[i] = force

        for i in range(n):
            v[i] += 0.5 * (a[i] + a_new[i]) * dt
            a[i] = a_new[i]

    # Compute total energy
    energy = 0.0
    for i in range(n):
        energy += 0.5 * v[i] * v[i]  # kinetic
        energy += 0.5 * k * x[i] * x[i]  # potential (self)
        if i < n - 1:
            dx = x[i + 1] - x[i]
            energy += 0.5 * k * dx * dx  # coupling potential

    return energy
