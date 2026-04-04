"""Compute total electrostatic potential energy of charges in 1D.

Keywords: physics, coulomb, electrostatic, potential, pairwise, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(3000,))
def coulomb_force(n: int) -> float:
    """Compute total electrostatic potential energy of n charges in 1D.

    Position x[i] = i*0.1, charge q[i] = (-1)^i.
    Sum all pairwise k*q1*q2/r where k = 8.9875e9.

    Args:
        n: Number of charges.

    Returns:
        Total potential energy as a float.
    """
    k = 8.9875e9
    total = 0.0

    for i in range(n):
        xi = i * 0.1
        qi = 1.0 if i % 2 == 0 else -1.0
        for j in range(i + 1, n):
            xj = j * 0.1
            qj = 1.0 if j % 2 == 0 else -1.0
            r = abs(xi - xj)
            if r > 0:
                total += k * qi * qj / r

    return total
