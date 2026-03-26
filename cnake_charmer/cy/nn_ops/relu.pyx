# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""ReLU on a C array tensor (basic Cython, scalar loop).

Same algorithm as Python but with cdef types and C array.
This is the baseline that SIMD should beat.

Keywords: relu, activation, neural network, tensor, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(10000000,))
def relu(int n):
    """Allocate C array tensor, apply ReLU in-place, return sum."""
    cdef int *data = <int *>malloc(n * sizeof(int))
    if not data:
        raise MemoryError()

    cdef int i
    cdef long long total = 0

    # Allocate tensor
    for i in range(n):
        data[i] = (i * 17 + 5) % 201 - 100

    # Apply ReLU in-place (scalar)
    for i in range(n):
        if data[i] < 0:
            data[i] = 0

    # Reduce
    for i in range(n):
        total += data[i]

    free(data)
    return int(total)
