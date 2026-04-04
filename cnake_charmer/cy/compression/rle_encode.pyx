# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Run-length encoding of a byte sequence (Cython-optimized).

Keywords: compression, RLE, run-length encoding, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(500000,))
def rle_encode(int n):
    """Run-length encode a byte sequence and return summary statistics.

    Input: byte[i] = (i // 7) & 0xFF — produces runs of length 7.

    Returns:
        Tuple of (num_runs, total_encoded_bytes, checksum % 10**9).
    """
    if n == 0:
        return (0, 0, 0)

    cdef unsigned char *data = <unsigned char *>malloc(n * sizeof(unsigned char))
    if not data:
        raise MemoryError()

    cdef int i
    cdef long long num_runs, checksum
    cdef unsigned char current
    cdef int run_len

    with nogil:
        for i in range(n):
            data[i] = (i // 7) & 0xFF

        num_runs = 0
        checksum = 0
        current = data[0]
        run_len = 1

        for i in range(1, n):
            if data[i] == current and run_len < 255:
                run_len += 1
            else:
                num_runs += 1
                checksum += run_len
                current = data[i]
                run_len = 1

        # Final run
        num_runs += 1
        checksum += run_len

    free(data)
    cdef long long total_encoded_bytes = 2 * num_runs
    return (num_runs, total_encoded_bytes, checksum % (10 ** 9))
