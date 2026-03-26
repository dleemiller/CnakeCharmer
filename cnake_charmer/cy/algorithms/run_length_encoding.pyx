# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Run-length encoding of a deterministic integer array.

Keywords: algorithms, run length encoding, compression, counting, cython, benchmark
"""

from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(10000000,))
def run_length_encoding(int n):
    """Count the number of runs in RLE of arr[i] = ((i*3+7)//5) % 10."""
    if n == 0:
        return 0

    cdef int runs = 1
    cdef int prev = ((0 * 3 + 7) // 5) % 10
    cdef int val, i

    for i in range(1, n):
        val = ((i * 3 + 7) // 5) % 10
        if val != prev:
            runs += 1
            prev = val

    return runs
