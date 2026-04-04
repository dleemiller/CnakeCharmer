"""Lotka-Volterra predator-prey simulation.

Keywords: simulation, predator, prey, Lotka-Volterra, ODE, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(5000,))
def predator_prey(n: int) -> float:
    """Simulate Lotka-Volterra predator-prey dynamics for n*100 steps.

    Uses Euler method with dt=0.001.
    Initial: prey=100.0, predator=20.0.
    Parameters: alpha=0.1, beta=0.002, gamma=0.4, delta=0.001.

    Args:
        n: Scale factor; total steps = n * 100.

    Returns:
        Final prey population as a float.
    """
    dt = 0.001
    steps = n * 100
    prey = 100.0
    predator = 20.0
    alpha = 0.1
    beta = 0.002
    gamma = 0.4
    delta = 0.001

    for _ in range(steps):
        dprey = (alpha * prey - beta * prey * predator) * dt
        dpredator = (delta * prey * predator - gamma * predator) * dt
        prey += dprey
        predator += dpredator

    return prey
