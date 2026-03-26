# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""
Maximum product subarray using dynamic programming (Cython-optimized).

Keywords: dynamic programming, max product, subarray, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(200000,))
def max_product_subarray(int n):
    """Find maximum product subarray across chunks using C arrays."""
    cdef int chunk_size = 10
    cdef int k = n / chunk_size
    cdef long long total = 0
    cdef int positive_count = 0
    cdef int chunk, i, offset
    cdef long long v, max_prod, min_prod, best
    cdef long long new_max, new_min, cand1, cand2
    cdef int *vals = <int *>malloc(chunk_size * sizeof(int))

    if vals == NULL:
        raise MemoryError("Failed to allocate array")

    for chunk in range(k):
        offset = chunk * chunk_size

        # Generate chunk values, map 0 -> 1
        for i in range(chunk_size):
            v = ((<long long>(offset + i) * 2654435761) % 11) - 5
            if v == 0:
                v = 1
            vals[i] = <int>v

        # Track max and min product
        max_prod = vals[0]
        min_prod = vals[0]
        best = vals[0]

        for i in range(1, chunk_size):
            v = vals[i]
            cand1 = max_prod * v
            cand2 = min_prod * v

            # new_max = max(v, cand1, cand2)
            new_max = v
            if cand1 > new_max:
                new_max = cand1
            if cand2 > new_max:
                new_max = cand2

            # new_min = min(v, cand1, cand2)
            new_min = v
            if cand1 < new_min:
                new_min = cand1
            if cand2 < new_min:
                new_min = cand2

            max_prod = new_max
            min_prod = new_min
            if max_prod > best:
                best = max_prod

        total += best
        if best > 0:
            positive_count += 1

    free(vals)
    return (total, positive_count)
