# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Apply gamma correction to an n x n deterministic grayscale image (Cython-optimized).

Keywords: image processing, gamma correction, brightness, power law, cython, benchmark
"""

from libc.math cimport pow
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(500,))
def gamma_correction(int n):
    """Apply gamma correction with three gamma values to an n x n image."""
    cdef int i, j, v, pixel, rounded
    cdef double inv_255 = 1.0 / 255.0
    cdef double corrected
    cdef long long sum0 = 0, sum1 = 0, sum2 = 0
    cdef double gammas[3]
    gammas[0] = 0.5
    gammas[1] = 1.5
    gammas[2] = 2.2

    # Build lookup tables
    cdef int lut0[256]
    cdef int lut1[256]
    cdef int lut2[256]

    for v in range(256):
        corrected = 255.0 * pow(v * inv_255, gammas[0])
        rounded = <int>(corrected + 0.5)
        if rounded < 0:
            rounded = 0
        elif rounded > 255:
            rounded = 255
        lut0[v] = rounded

        corrected = 255.0 * pow(v * inv_255, gammas[1])
        rounded = <int>(corrected + 0.5)
        if rounded < 0:
            rounded = 0
        elif rounded > 255:
            rounded = 255
        lut1[v] = rounded

        corrected = 255.0 * pow(v * inv_255, gammas[2])
        rounded = <int>(corrected + 0.5)
        if rounded < 0:
            rounded = 0
        elif rounded > 255:
            rounded = 255
        lut2[v] = rounded

    # Generate image and apply each gamma LUT
    for i in range(n):
        for j in range(n):
            pixel = (i * 17 + j * 31) % 256
            sum0 += lut0[pixel]
            sum1 += lut1[pixel]
            sum2 += lut2[pixel]

    return (<int>sum0, <int>sum1, <int>sum2)
