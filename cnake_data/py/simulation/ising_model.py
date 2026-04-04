"""1D Ising model with Metropolis algorithm.

Keywords: simulation, ising model, metropolis, statistical mechanics, spin, benchmark
"""

import math

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(2000,))
def ising_model(n: int) -> float:
    """Simulate 1D Ising model with Metropolis updates over 1000 sweeps.

    n spins initialized as: spin[i] = 1 if (i*7+3)%2 == 0 else -1.
    Coupling J=1.0, temperature T=2.0.
    Deterministic RNG: rand for site i at sweep s = ((i*2654435761 + s*1013904223) % (2^32)) / 2^32.

    Each sweep updates all n spins sequentially.

    Args:
        n: Number of spins.

    Returns:
        Final total energy E = -J * sum(s[i]*s[i+1]) for periodic boundary.
    """
    sweeps = 1000
    coupling = 1.0
    temp = 2.0
    mod = 2**32

    spin = [0] * n
    for i in range(n):
        if (i * 7 + 3) % 2 == 0:
            spin[i] = 1
        else:
            spin[i] = -1

    for s in range(sweeps):
        for i in range(n):
            left = spin[(i - 1) % n]
            right = spin[(i + 1) % n]
            de = 2.0 * coupling * spin[i] * (left + right)
            if de <= 0:
                spin[i] = -spin[i]
            else:
                rand_val = ((i * 2654435761 + s * 1013904223) % mod) / mod
                if rand_val < math.exp(-de / temp):
                    spin[i] = -spin[i]

    energy = 0.0
    for i in range(n):
        energy -= coupling * spin[i] * spin[(i + 1) % n]
    return energy
