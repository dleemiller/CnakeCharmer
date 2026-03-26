# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""1D Ising model with Metropolis algorithm (Cython-optimized).

Keywords: simulation, ising model, metropolis, statistical mechanics, spin, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from libc.math cimport exp
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(2000,))
def ising_model(int n):
    """Simulate 1D Ising model with Metropolis updates using C arrays."""
    cdef int sweeps = 1000
    cdef double coupling = 1.0
    cdef double temp = 2.0
    cdef int i, s
    cdef int left_spin, right_spin
    cdef double de, rand_val, energy
    cdef unsigned int hash_val

    cdef int *spin = <int *>malloc(n * sizeof(int))
    if not spin:
        raise MemoryError()

    for i in range(n):
        if (i * 7 + 3) % 2 == 0:
            spin[i] = 1
        else:
            spin[i] = -1

    for s in range(sweeps):
        for i in range(n):
            left_spin = spin[(i - 1 + n) % n]
            right_spin = spin[(i + 1) % n]
            de = 2.0 * coupling * spin[i] * (left_spin + right_spin)
            if de <= 0:
                spin[i] = -spin[i]
            else:
                hash_val = (<unsigned int>i * <unsigned int>2654435761U + <unsigned int>s * <unsigned int>1013904223U)
                rand_val = <double>hash_val / 4294967296.0
                if rand_val < exp(-de / temp):
                    spin[i] = -spin[i]

    energy = 0.0
    for i in range(n):
        energy -= coupling * spin[i] * spin[(i + 1) % n]

    free(spin)
    return energy
