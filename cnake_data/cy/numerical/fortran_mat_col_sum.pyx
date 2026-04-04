# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Column sums of a Fortran-contiguous 2D memoryview.

Allocates a column-major matrix via cvarray, fills it with
hash-derived doubles, then sums each column.

Keywords: numerical, matrix, column sum, fortran, memoryview, cython, benchmark
"""

from cython.view cimport array as cvarray
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(300,))
def fortran_mat_col_sum(int n):
    """Compute sum of all column sums of an n x n matrix."""
    cdef int i, j, idx
    cdef unsigned int h
    cdef double col_sum, total

    arr = cvarray(
        shape=(n, n),
        itemsize=sizeof(double),
        format="d",
        mode="fortran",
    )
    cdef double[::1, :] view = arr

    for j in range(n):
        for i in range(n):
            idx = j * n + i
            h = (
                (<unsigned int>idx
                 * <unsigned int>2654435761)
                ^ (<unsigned int>idx
                   * <unsigned int>2246822519)
            )
            view[i, j] = (
                <double>(h & 0xFFFF) / 65535.0
            )

    total = 0.0
    for j in range(n):
        col_sum = 0.0
        for i in range(n):
            col_sum += view[i, j]
        total += col_sum
    return total
