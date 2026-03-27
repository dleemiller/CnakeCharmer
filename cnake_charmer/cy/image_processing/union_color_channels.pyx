# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Pack/unpack pixel values using union with byte array and
compute channel averages (Cython-optimized).

Keywords: union, color, channels, image processing, cython, benchmark
"""

from cnake_charmer.benchmarks import cython_benchmark

cdef union ColorUnion:
    unsigned int packed
    unsigned char[4] bytes


@cython_benchmark(syntax="cy", args=(100000,))
def union_color_channels(int n):
    """Pack n pixels, extract channels via union, averages."""
    cdef long long r_sum = 0
    cdef long long g_sum = 0
    cdef int i
    cdef unsigned long long h
    cdef ColorUnion cu

    for i in range(n):
        h = (<unsigned long long>i
             * <unsigned long long>2654435761)
        cu.packed = <unsigned int>(
            h & <unsigned long long>0xFFFFFFFF
        )
        r_sum += cu.bytes[0]
        g_sum += cu.bytes[1]
    cdef int r_avg = <int>(r_sum / n)
    cdef int g_avg = <int>(g_sum / n)
    return r_avg * 1000 + g_avg
