# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""
Compute the Collatz sequence length for each number from 1 to n (Cython-optimized).

Keywords: collatz, sequence, math, conjecture, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(10000,))
def collatz_lengths(int n):
    """Compute Collatz sequence lengths using C-typed loop and C array."""
    cdef int i, count
    cdef long long val
    cdef int *arr = <int *>malloc(n * sizeof(int))

    if arr == NULL:
        raise MemoryError("Failed to allocate array")

    for i in range(n):
        count = 0
        val = i + 1
        while val != 1:
            if val % 2 == 0:
                val = val // 2
            else:
                val = 3 * val + 1
            count += 1
        arr[i] = count

    cdef list result = [arr[i] for i in range(n)]
    free(arr)
    return result
