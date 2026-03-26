# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Run-length encode a deterministic string and count runs (Cython-optimized).

Keywords: string processing, run-length encoding, compression, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(1000000,))
def run_length_encode(int n):
    """Count runs by iterating over a C char array directly."""
    if n == 0:
        return (0, 0)

    cdef char *chars = <char *>malloc(n * sizeof(char))
    if not chars:
        raise MemoryError()

    cdef int i
    cdef int runs = 1
    cdef int current_run_len = 1
    cdef int max_run_len = 1

    # Build the character array: 65 + (i*3)%5 maps to A-E
    for i in range(n):
        chars[i] = 65 + (i * 3) % 5

    for i in range(1, n):
        if chars[i] != chars[i - 1]:
            runs += 1
            current_run_len = 1
        else:
            current_run_len += 1
            if current_run_len > max_run_len:
                max_run_len = current_run_len

    cdef int result_runs = runs
    cdef int result_max = max_run_len
    free(chars)
    return (result_runs, result_max)
