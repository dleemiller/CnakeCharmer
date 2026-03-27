# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Process pixel data by color channel using cpdef enum (Cython-optimized).

Keywords: image processing, color, channel, cpdef enum, pixel, blend, cython, benchmark
"""

from cnake_charmer.benchmarks import cython_benchmark

cpdef enum Channel:
    RED = 0
    GREEN = 1
    BLUE = 2


cdef int _transform_channel(int value, int channel) noexcept:
    cdef int result
    if channel == RED:
        result = <int>(value * 0.8)
    elif channel == GREEN:
        result = <int>(value * 1.1)
        if result > 255:
            result = 255
    else:
        result = <int>(value * 0.9)
    return result


@cython_benchmark(syntax="cy", args=(100000,))
def cpdef_enum_color_blend(int n):
    """Process pixels by channel using cpdef enum dispatch."""
    cdef int i, r, g, b
    cdef long long total = 0

    for i in range(n):
        r = (i * 41 + 7) % 256
        g = (i * 59 + 13) % 256
        b = (i * 71 + 3) % 256

        total += _transform_channel(r, RED)
        total += _transform_channel(g, GREEN)
        total += _transform_channel(b, BLUE)

    return <int>total
