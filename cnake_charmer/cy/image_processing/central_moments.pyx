# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Compute central moments of a 2D image up to order 3 (Cython-optimized).

Keywords: image processing, central moments, statistics, moment computation, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(80,))
def central_moments(int n):
    """Compute central image moments mu_pq up to order 3 on an n x n image."""
    cdef int nn = n * n
    cdef double *img = <double *>malloc(nn * sizeof(double))
    if not img:
        raise MemoryError()

    cdef int r, c, p
    cdef double val, dr, dc
    cdef double sum_val = 0.0
    cdef double sum_r = 0.0
    cdef double sum_c = 0.0
    cdef double cr_val, cc_val
    # 4x4 moment array stored flat (16 entries)
    cdef double mu[16]

    # Generate deterministic image
    for r in range(n):
        for c in range(n):
            img[r * n + c] = <double>((r * 7 + c * 13 + 42) % 256)

    # Compute centroid
    with nogil:
        for r in range(n):
            for c in range(n):
                val = img[r * n + c]
                sum_val += val
                sum_r += r * val
                sum_c += c * val

        cr_val = sum_r / sum_val
        cc_val = sum_c / sum_val

        # Zero moment array
        for p in range(16):
            mu[p] = 0.0

        # Compute central moments up to order 3
        for r in range(n):
            for c in range(n):
                val = img[r * n + c]
                dr = r - cr_val
                dc = c - cc_val

                # Manually unroll p,q combinations where p+q <= 3
                # mu[p*4 + q]
                # p=0, q=0
                mu[0] += val
                # p=0, q=1
                mu[1] += val * dr
                # p=0, q=2
                mu[2] += val * dr * dr
                # p=0, q=3
                mu[3] += val * dr * dr * dr
                # p=1, q=0
                mu[4] += val * dc
                # p=1, q=1
                mu[5] += val * dc * dr
                # p=1, q=2
                mu[6] += val * dc * dr * dr
                # p=2, q=0
                mu[8] += val * dc * dc
                # p=2, q=1
                mu[9] += val * dc * dc * dr
                # p=3, q=0
                mu[12] += val * dc * dc * dc

    free(img)

    # Return (mu_20, mu_02, mu_11, mu_00)
    return (mu[8], mu[2], mu[5], mu[0])
