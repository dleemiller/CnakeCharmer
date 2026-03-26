"""SIR epidemic model simulation.

Keywords: SIR model, epidemic, simulation, population dynamics, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(100000,))
def epidemic_sir(n: int) -> int:
    """Simulate SIR epidemic model.

    Population size n, initial: S=n-10, I=10, R=0.
    Beta=0.3/n, gamma=0.1, dt=0.1, 1000 timesteps.

    Args:
        n: Population size.

    Returns:
        Final recovered count as int.
    """
    s = float(n - 10)
    inf = 10.0
    r = 0.0
    beta = 0.3 / n
    gamma = 0.1
    dt = 0.1

    for _t in range(1000):
        new_infections = beta * s * inf * dt
        new_recoveries = gamma * inf * dt
        s -= new_infections
        inf += new_infections - new_recoveries
        r += new_recoveries

    return int(r)
