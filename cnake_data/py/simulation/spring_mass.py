"""1D spring-mass chain simulation.

Keywords: simulation, spring, mass, physics, Hooke's law, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(3000,))
def spring_mass(n: int) -> float:
    """Simulate n masses connected by springs in 1D for 1000 timesteps.

    Initial positions: x[i] = i * 1.0
    Initial velocities: v[i] = 0.0
    Spring constant k = 1.0, timestep dt = 0.01.
    Forces from Hooke's law between adjacent masses.
    Returns sum of final positions.

    Args:
        n: Number of masses.

    Returns:
        Sum of final positions.
    """
    timesteps = 1000
    k = 1.0
    dt = 0.01

    x = [i * 1.0 for i in range(n)]
    v = [0.0] * n

    for _ in range(timesteps):
        # Compute forces
        for i in range(n):
            force = 0.0
            if i > 0:
                force += k * (x[i - 1] - x[i])
            if i < n - 1:
                force += k * (x[i + 1] - x[i])
            v[i] += force * dt
        # Update positions
        for i in range(n):
            x[i] += v[i] * dt

    total = 0.0
    for i in range(n):
        total += x[i]
    return total
