# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Maximum sum submatrix using 2D Kadane's algorithm.

Keywords: algorithms, kadane, 2d, max submatrix, dynamic programming, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from libc.string cimport memset
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(200,))
def kadane_2d(int n):
    """Find the maximum sum submatrix in an n x n matrix."""
    cdef int *matrix = <int *>malloc(n * n * sizeof(int))
    cdef int *col_sum = <int *>malloc(n * sizeof(int))
    if not matrix or not col_sum:
        if matrix: free(matrix)
        if col_sum: free(col_sum)
        raise MemoryError()

    cdef int i, j, left, right, row
    cdef long long current, max_here, best

    # Build matrix
    for i in range(n):
        for j in range(n):
            matrix[i * n + j] = ((i * 7 + j * 13) % 201) - 100

    best = matrix[0]

    for left in range(n):
        memset(col_sum, 0, n * sizeof(int))
        for right in range(left, n):
            for row in range(n):
                col_sum[row] += matrix[row * n + right]

            # Kadane on col_sum
            current = col_sum[0]
            max_here = col_sum[0]
            for row in range(1, n):
                if current + col_sum[row] > col_sum[row]:
                    current = current + col_sum[row]
                else:
                    current = col_sum[row]
                if current > max_here:
                    max_here = current
            if max_here > best:
                best = max_here

    free(matrix)
    free(col_sum)
    return int(best)
