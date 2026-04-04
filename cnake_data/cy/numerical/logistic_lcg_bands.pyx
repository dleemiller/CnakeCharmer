# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Run a fixed-point logistic map and summarize occupancy bands (Cython)."""

from cnake_data.benchmarks import cython_benchmark


cdef void _logistic_kernel(
    int n,
    int scale,
    int r,
    int* bins0_out,
    int* bins3_out,
    int* bins7_out,
    unsigned int* checksum_out,
) noexcept nogil:
    cdef int x = 123456
    cdef int i, b
    cdef int bins0 = 0
    cdef int bins3 = 0
    cdef int bins7 = 0
    cdef unsigned int checksum = 0
    cdef unsigned int mask = 0xFFFFFFFF

    for i in range(n):
        x = <int>((<long long>r * x * (scale - x)) // (<long long>scale * scale))
        b = (x * 8) // scale
        if b > 7:
            b = 7

        if b == 0:
            bins0 += 1
        elif b == 3:
            bins3 += 1
        elif b == 7:
            bins7 += 1

        checksum = (checksum + <unsigned int>(x ^ (i * 1315423911))) & mask

    bins0_out[0] = bins0
    bins3_out[0] = bins3
    bins7_out[0] = bins7
    checksum_out[0] = checksum


@cython_benchmark(syntax="cy", args=(300000,))
def logistic_lcg_bands(int n):
    cdef int scale = 1 << 20
    cdef int r = (39 * (1 << 20)) // 10
    cdef int bins0 = 0
    cdef int bins3 = 0
    cdef int bins7 = 0
    cdef unsigned int checksum = 0

    with nogil:
        _logistic_kernel(n, scale, r, &bins0, &bins3, &bins7, &checksum)

    return (bins0, bins3, bins7, checksum)
