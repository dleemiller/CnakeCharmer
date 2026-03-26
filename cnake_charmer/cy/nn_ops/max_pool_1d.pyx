# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""1D max pooling.

Keywords: max pool, pooling, neural network, downsampling, cython
"""

from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(3000000,))
def max_pool_1d(int n):
    """Max pool with kernel=4, stride=4 and return sum of pooled values."""
    cdef long long total = 0
    cdef int num_pools = n // 4
    cdef int i, j, base, max_val, v

    for i in range(num_pools):
        base = i * 4
        max_val = (base * 31 + 17) % 1000
        for j in range(1, 4):
            v = ((base + j) * 31 + 17) % 1000
            if v > max_val:
                max_val = v
        total += max_val
    return int(total)
