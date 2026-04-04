# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Generate a spiral matrix and compute statistics (Cython-optimized).

Keywords: leetcode, spiral matrix, generation, traversal, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(1500,))
def spiral_matrix(int n):
    """Generate spiral matrix using C array and typed loops."""
    cdef int total = n * n
    cdef int *mat = <int *>malloc(total * sizeof(int))
    if not mat:
        raise MemoryError()

    cdef int top = 0
    cdef int bottom = n - 1
    cdef int left = 0
    cdef int right = n - 1
    cdef int val = 1
    cdef int row, col, i

    while top <= bottom and left <= right:
        for col in range(left, right + 1):
            mat[top * n + col] = val
            val += 1
        top += 1

        for row in range(top, bottom + 1):
            mat[row * n + right] = val
            val += 1
        right -= 1

        if top <= bottom:
            for col in range(right, left - 1, -1):
                mat[bottom * n + col] = val
                val += 1
            bottom -= 1

        if left <= right:
            for row in range(bottom, top - 1, -1):
                mat[row * n + left] = val
                val += 1
            left += 1

    cdef long long corner_sum = mat[0] + mat[n - 1] + mat[(n - 1) * n] + mat[(n - 1) * n + n - 1]
    cdef int center_val = mat[(n // 2) * n + n // 2]

    cdef long long diagonal_sum = 0
    for i in range(n):
        diagonal_sum += mat[i * n + i]

    free(mat)
    return (int(corner_sum), center_val, int(diagonal_sum))
