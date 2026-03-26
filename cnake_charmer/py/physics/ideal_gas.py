"""Compute pressure for ideal gas states using PV=nRT.

Keywords: physics, ideal, gas, pressure, thermodynamics, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(5000000,))
def ideal_gas(n: int) -> float:
    """Compute pressure for n ideal gas states using PV=nRT.

    V[i] = (i%100 + 1) * 0.001 m^3, T[i] = (i%500 + 200) K, n_mol=1.
    Returns sum of all pressures.

    Args:
        n: Number of gas states.

    Returns:
        Sum of pressures as a float.
    """
    R = 8.314
    n_mol = 1.0

    total = 0.0
    for i in range(n):
        V = (i % 100 + 1) * 0.001
        T = i % 500 + 200
        P = n_mol * R * T / V
        total += P

    return total
