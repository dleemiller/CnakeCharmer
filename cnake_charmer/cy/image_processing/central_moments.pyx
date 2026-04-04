# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Raw image moments up to order 3 for a deterministic n×n image (Cython-optimized).

Keywords: image processing, moments, raw moments, statistics, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(300,))
def central_moments(int n):
    """Compute raw image moments M[p][q] for p+q <= 3 on an n×n image.

    Args:
        n: Image side length (n×n pixels).

    Returns:
        Tuple of (M[0][0], M[1][0], M[0][1], M[2][0]) as integers.
    """
    cdef int r, c, pixel
    cdef long long m00 = 0, m10 = 0, m01 = 0, m20 = 0
    cdef int *img = <int *>malloc(n * n * sizeof(int))
    if not img:
        raise MemoryError()

    with nogil:
        for r in range(n):
            for c in range(n):
                img[r * n + c] = (r * 1009 + c * 2003 + 42) % 256

        for r in range(n):
            for c in range(n):
                pixel = img[r * n + c]
                m00 += pixel
                m10 += r * pixel
                m01 += c * pixel
                m20 += r * r * pixel

    free(img)
    return (int(m00), int(m10), int(m01), int(m20))
