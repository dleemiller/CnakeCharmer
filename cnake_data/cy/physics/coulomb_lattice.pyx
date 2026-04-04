# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Total electrostatic energy and force for n charges on a line (Cython-optimized).

Keywords: physics, electrostatics, coulomb, lattice, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(2000,))
def coulomb_lattice(int n):
    """Compute Coulomb energy and force on charge 0 for n alternating charges.

    Charges equally spaced (unit sep), alternating +1/-1.
    """
    cdef double *pos = <double *>malloc(n * sizeof(double))
    cdef double *q = <double *>malloc(n * sizeof(double))
    if not pos or not q:
        if pos: free(pos)
        if q: free(q)
        raise MemoryError()

    cdef int i, j
    cdef double U, F, r, q0
    cdef long long iU, iF

    with nogil:
        for i in range(n):
            pos[i] = <double>i
            q[i] = 1.0 if (i % 2 == 0) else -1.0

        U = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                r = pos[j] - pos[i]  # always positive since j > i
                U += q[i] * q[j] / r

        F = 0.0
        q0 = q[0]
        for j in range(1, n):
            r = pos[j] - pos[0]  # always positive
            # sign is +1 since r > 0
            F += q0 * q[j] / (r * r)

    iU = <long long>(U * 1e6)
    iF = <long long>(F * 1e9)

    free(pos)
    free(q)
    return (iU, iF)
