# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Bilinear interpolation upsampling of a grayscale image (Cython-optimized).

Keywords: image processing, bilinear, interpolation, upsampling, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(300,))
def bilinear_interpolation(int n):
    """Upsample n x n image to (2n-1) x (2n-1) using bilinear interpolation."""
    cdef int i, j, oi, oj, i0, j0, i1, j1, out_n
    cdef double si, sj, fi, fj, val
    cdef long long total

    cdef int *src = <int *>malloc(n * n * sizeof(int))
    if not src:
        raise MemoryError()

    # Generate source image
    for i in range(n):
        for j in range(n):
            src[i * n + j] = (i * 7 + j * 13 + 3) % 256

    out_n = 2 * n - 1
    total = 0

    for oi in range(out_n):
        for oj in range(out_n):
            si = oi / 2.0
            sj = oj / 2.0

            i0 = <int>si
            j0 = <int>sj
            i1 = i0 + 1
            if i1 >= n:
                i1 = n - 1
            j1 = j0 + 1
            if j1 >= n:
                j1 = n - 1

            fi = si - i0
            fj = sj - j0

            val = (src[i0 * n + j0] * (1.0 - fi) * (1.0 - fj) +
                   src[i1 * n + j0] * fi * (1.0 - fj) +
                   src[i0 * n + j1] * (1.0 - fi) * fj +
                   src[i1 * n + j1] * fi * fj)

            total += <int>(val + 0.5)

    free(src)
    return total
