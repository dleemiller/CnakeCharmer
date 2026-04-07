# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""In-place L2 row normalization — Cython implementation."""

from libc.math cimport sin, sqrt
from libc.stdlib cimport free, malloc

from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(300, 200))
def row_l2_normalize(int n, int m):
    """L2-normalize each row of a deterministically generated n×m matrix.

    Args:
        n: Number of rows.
        m: Number of columns.

    Returns:
        Sum of all squared elements after normalization (≈ number of non-zero rows).
    """
    cdef int i, j
    cdef double norm_sq, norm, val, total
    cdef double *mat = <double *>malloc(n * m * sizeof(double))
    if not mat:
        raise MemoryError()

    # Generate deterministic matrix values
    for i in range(n):
        for j in range(m):
            mat[i * m + j] = sin(i * 1.7 + j * 0.9)

    # Normalize each row in-place
    for i in range(n):
        norm_sq = 0.0
        for j in range(m):
            val = mat[i * m + j]
            norm_sq += val * val
        if norm_sq == 0.0:
            continue
        norm = sqrt(norm_sq)
        for j in range(m):
            mat[i * m + j] /= norm

    # Checksum
    total = 0.0
    for i in range(n):
        for j in range(m):
            val = mat[i * m + j]
            total += val * val

    free(mat)
    return total
