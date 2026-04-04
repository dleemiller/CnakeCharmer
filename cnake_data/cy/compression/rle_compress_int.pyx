# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Run-length encode integers and count runs (Cython-optimized).

Keywords: run-length encoding, RLE, compression, cython, benchmark
"""

from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(3000000,))
def rle_compress_int(int n):
    """Run-length encode n integers using pure typed loop."""
    if n == 0:
        return 0

    cdef int i, runs, prev, curr

    runs = 1
    prev = ((0 * 37 + 7) // 4) % 10

    for i in range(1, n):
        curr = ((i * 37 + 7) // 4) % 10
        if curr != prev:
            runs += 1
            prev = curr

    return runs
