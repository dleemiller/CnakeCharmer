# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""ReLU activation function (basic Cython, no SIMD).

Keywords: relu, activation, neural network, elementwise, cython
"""

from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(10000000,))
def relu(int n):
    """Apply ReLU to n values and return sum of activated values."""
    cdef long long total = 0
    cdef int i, v

    for i in range(n):
        v = (i * 17 + 5) % 201 - 100
        if v > 0:
            total += v
    return int(total)
