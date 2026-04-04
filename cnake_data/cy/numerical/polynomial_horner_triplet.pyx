# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Evaluate cubic polynomials over deterministic inputs (Cython)."""

from cnake_data.benchmarks import cython_benchmark


cdef void _poly_kernel(
    int n,
    int mod,
    int* total_out,
    unsigned int* xor_out,
    int* max_out,
) noexcept nogil:
    cdef int i, v
    cdef long long p
    cdef int total = 0
    cdef unsigned int alt_xor = 0
    cdef unsigned int mask = 0xFFFFFFFF
    cdef int max_val = 0

    for i in range(n):
        v = ((i * 37 + 11) % 1009) - 504
        p = (((3 * v - 2) * v + 5) * v - 7) % mod
        if p < 0:
            p += mod
        total = (total + <int>p) % mod
        alt_xor ^= <unsigned int>((p + i) & mask)
        if p > max_val:
            max_val = <int>p

    total_out[0] = total
    xor_out[0] = alt_xor
    max_out[0] = max_val


@cython_benchmark(syntax="cy", args=(200000,))
def polynomial_horner_triplet(int n):
    cdef int mod = 1000003
    cdef int total = 0
    cdef unsigned int alt_xor = 0
    cdef int max_val = 0

    with nogil:
        _poly_kernel(n, mod, &total, &alt_xor, &max_val)

    return (total, alt_xor, max_val)
