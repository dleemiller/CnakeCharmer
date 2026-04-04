# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Count total set bits across all integers in a range.

Keywords: algorithms, bit counting, popcount, bitwise, range, cython, benchmark
"""

from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(2000000,))
def bit_count_range(int n):
    """Count set bits across all integers in [0, n) and compute related statistics."""
    cdef long long total_bits = 0
    cdef int count_with_10 = 0
    cdef int max_odd_run = 0
    cdef int current_odd_run = 0
    cdef int i, count
    cdef unsigned int x

    for i in range(n):
        # Count set bits
        x = <unsigned int>i
        count = 0
        while x:
            count += x & 1
            x >>= 1

        total_bits += count

        if count == 10:
            count_with_10 += 1

        if count & 1:
            current_odd_run += 1
            if current_odd_run > max_odd_run:
                max_odd_run = current_odd_run
        else:
            current_odd_run = 0

    return (int(total_bits), count_with_10, max_odd_run)
