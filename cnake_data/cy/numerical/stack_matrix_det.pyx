# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Compute 3x3 matrix determinants using stack C array.

Keywords: numerical, matrix, determinant, stack array, cython, benchmark
"""

from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(100000,))
def stack_matrix_det(int n):
    """Compute determinants of n 3x3 matrices."""
    cdef double mat[9]
    cdef double total = 0.0
    cdef double det
    cdef long long seed
    cdef int k, i

    for k in range(n):
        seed = (
            <long long>k * <long long>2654435761 + 17
        )
        for i in range(9):
            seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
            mat[i] = (seed % 10000) / 1000.0 - 5.0

        det = (
            mat[0] * (mat[4] * mat[8] - mat[5] * mat[7])
            - mat[1] * (mat[3] * mat[8] - mat[5] * mat[6])
            + mat[2] * (mat[3] * mat[7] - mat[4] * mat[6])
        )
        total += det

    return total
