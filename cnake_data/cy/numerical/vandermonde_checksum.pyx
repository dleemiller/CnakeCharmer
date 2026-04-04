# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Compute summary checksums from a small Vandermonde-like transform (Cython)."""

from cnake_data.benchmarks import cython_benchmark


cdef void _vandermonde_kernel(
    int n,
    int mod,
    int* sum_first_out,
    int* sum_last_out,
    unsigned int* diag_xor_out,
) noexcept nogil:
    cdef int i, j
    cdef long long power
    cdef int base
    cdef int sum_first = 0
    cdef int sum_last = 0
    cdef unsigned int diag_xor = 0
    cdef unsigned int mask = 0xFFFFFFFF
    cdef int row_sum

    for i in range(n):
        base = (i * 13 + 7) % 257
        power = 1
        row_sum = 0
        for j in range(8):
            row_sum = (row_sum + <int>power) % mod
            if j == 7:
                sum_last = (sum_last + <int>power) % mod
            power = (power * base + j + 1) % mod
        sum_first = (sum_first + 1) % mod
        diag_xor ^= <unsigned int>((row_sum + i * 17) & mask)

    sum_first_out[0] = sum_first
    sum_last_out[0] = sum_last
    diag_xor_out[0] = diag_xor


@cython_benchmark(syntax="cy", args=(100000,))
def vandermonde_checksum(int n):
    cdef int mod = 1000000007
    cdef int sum_first = 0
    cdef int sum_last = 0
    cdef unsigned int diag_xor = 0

    with nogil:
        _vandermonde_kernel(n, mod, &sum_first, &sum_last, &diag_xor)

    return (sum_first, sum_last, diag_xor)
