"""
Total electrostatic energy and net force for n charges arranged in a line.

Keywords: physics, electrostatics, coulomb, lattice, benchmark
"""

import math

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(2000,))
def coulomb_lattice(n: int) -> tuple:
    """Compute total electrostatic energy and force on charge 0 for n charges.

    Charges are equally spaced (unit separation), alternating +1/-1.
    Energy: U = sum_{i<j} q[i]*q[j] / |pos[j]-pos[i]|
    Force on 0: F = sum_{j>0} q[0]*q[j] / (pos[j]-pos[0])^2 * sign(pos[j]-pos[0])

    Args:
        n: Number of charges.

    Returns:
        Tuple of (int(U * 1e6), int(F * 1e9)) for exact comparison.
    """
    pos = [float(i) for i in range(n)]
    q = [1.0 if i % 2 == 0 else -1.0 for i in range(n)]

    U = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            r = pos[j] - pos[i]  # always positive since j > i
            U += q[i] * q[j] / r

    F = 0.0
    q0 = q[0]
    for j in range(1, n):
        r = pos[j] - pos[0]  # always positive
        sign = math.copysign(1.0, r)
        F += q0 * q[j] / (r * r) * sign

    return (int(U * 1e6), int(F * 1e9))
