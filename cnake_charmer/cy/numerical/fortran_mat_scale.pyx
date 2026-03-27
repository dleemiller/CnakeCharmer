# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Scale columns of a Fortran-contiguous 2D memoryview.

Allocates column-major matrix via cvarray, fills with hash
doubles, scales each column by (j+1)/n, returns total sum.

Keywords: numerical, matrix, scale, fortran, memoryview, cython, benchmark
"""

from cython.view cimport array as cvarray
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(300,))
def fortran_mat_scale(int n):
    """Scale columns of hash-filled matrix, return sum."""
    cdef int i, j, idx
    cdef unsigned int h
    cdef double factor, total

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

    for j in range(n):
        factor = <double>(j + 1) / <double>n
        for i in range(n):
            view[i, j] *= factor

    total = 0.0
    for i in range(n):
        for j in range(n):
            total += view[i, j]
    return total
