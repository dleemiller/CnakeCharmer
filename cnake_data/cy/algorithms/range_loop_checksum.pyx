# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Accumulate checksum from mixed range-loop patterns (Cython)."""

from cnake_data.benchmarks import cython_benchmark


cdef void _range_loop_impl(
    int a,
    int b,
    int step,
    int factor,
    int rounds,
    int *checksum_out,
    int *last_i_out,
) noexcept nogil:
    cdef int checksum = 0
    cdef int last_i = 0
    cdef int r, i

    for r in range(rounds):
        for i in range(a):
            checksum += i + (r & 3)
            last_i = i

        for i in range(a, b):
            checksum += (i * factor) ^ (r & 7)
            last_i = i

        i = 0
        while i < b:
            checksum += i * (step + 1)
            i += step
        last_i = i

    checksum_out[0] = checksum
    last_i_out[0] = last_i


@cython_benchmark(syntax="cy", args=(12, 180, 3, 2, 90000))
def range_loop_checksum(int a, int b, int step, int factor, int rounds):
    cdef int checksum
    cdef int last_i = 0

    with nogil:
        _range_loop_impl(a, b, step, factor, rounds, &checksum, &last_i)

    return (checksum & 0xFFFFFFFF, last_i, rounds)
