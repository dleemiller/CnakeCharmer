# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Simulate a deterministic 2D lattice walk and summarize moments (Cython)."""

from cnake_data.benchmarks import cython_benchmark


cdef void _walk_kernel(
    int n,
    int* x_out,
    int* y_out,
    long long* max_d2_out,
    unsigned int* checksum_out,
) noexcept nogil:
    cdef int i
    cdef int x = 0
    cdef int y = 0
    cdef unsigned int state = 2463534242
    cdef unsigned int mask = 0xFFFFFFFF
    cdef unsigned int step
    cdef long long d2
    cdef long long max_d2 = 0
    cdef unsigned int checksum = 0

    for i in range(n):
        state = (1664525 * state + 1013904223) & mask
        step = state & 3
        if step == 0:
            x += 1
        elif step == 1:
            x -= 1
        elif step == 2:
            y += 1
        else:
            y -= 1

        d2 = x * x + y * y
        if d2 > max_d2:
            max_d2 = d2
        checksum = (checksum + (((x & 0xFFFF) << 16) + (y & 0xFFFF) + i)) & mask

    x_out[0] = x
    y_out[0] = y
    max_d2_out[0] = max_d2
    checksum_out[0] = checksum


@cython_benchmark(syntax="cy", args=(300000,))
def lattice_walk_moments(int n):
    cdef int x = 0
    cdef int y = 0
    cdef long long max_d2 = 0
    cdef unsigned int checksum = 0

    with nogil:
        _walk_kernel(n, &x, &y, &max_d2, &checksum)

    return (x, y, max_d2, checksum)
