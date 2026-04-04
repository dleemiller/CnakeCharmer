# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Submatrix correlation score for synthetic biclusters (Cython)."""

from libc.stdlib cimport malloc, free
from libc.math cimport sqrt, fabs
from cnake_data.benchmarks import cython_benchmark

cdef unsigned int MASK32 = 0xFFFFFFFF


cdef inline double pearson(double *a, double *b, int stride_a, int stride_b, int n) noexcept nogil:
    cdef int i
    cdef double sx = 0.0
    cdef double sy = 0.0
    cdef double sx2 = 0.0
    cdef double sy2 = 0.0
    cdef double sxy = 0.0
    cdef double xv, yv, num, den1, den2
    for i in range(n):
        xv = a[i * stride_a]
        yv = b[i * stride_b]
        sx += xv; sy += yv; sx2 += xv * xv; sy2 += yv * yv; sxy += xv * yv
    num = n * sxy - sx * sy
    den1 = n * sx2 - sx * sx
    den2 = n * sy2 - sy * sy
    if den1 <= 1e-12 or den2 <= 1e-12:
        return 0.0
    return num / sqrt(den1 * den2)


@cython_benchmark(syntax="cy", args=(72, 56, 19))
def bicluster_scs_score(int n_rows, int n_cols, int seed):
    cdef double *mat = <double *>malloc(n_rows * n_cols * sizeof(double))
    cdef unsigned int state
    cdef int i, j, i2, j2
    cdef double min_row_score = 2.0
    cdef double min_col_score = 2.0
    cdef double t, v, row_score, col_score

    if mat == NULL:
        raise MemoryError()

    state = <unsigned int>((seed * 1103515245 + 12345) & MASK32)
    for i in range(n_rows):
        for j in range(n_cols):
            state = (state * 1103515245 + 12345) & MASK32
            mat[i * n_cols + j] = ((<int>(state % 4096)) - 2048) / 157.0 + 0.03 * i + 0.02 * j

    for i in range(n_rows):
        t = 0.0
        for i2 in range(n_rows):
            if i != i2:
                v = pearson(&mat[i * n_cols], &mat[i2 * n_cols], 1, 1, n_cols)
                t += fabs(v)
        row_score = 1.0 - t / (n_rows - 1)
        if row_score < min_row_score:
            min_row_score = row_score

    for j in range(n_cols):
        t = 0.0
        for j2 in range(n_cols):
            if j != j2:
                v = pearson(&mat[j], &mat[j2], n_cols, n_cols, n_rows)
                t += fabs(v)
        col_score = 1.0 - t / (n_cols - 1)
        if col_score < min_col_score:
            min_col_score = col_score

    free(mat)
    return (min_row_score if min_row_score < min_col_score else min_col_score, min_row_score, min_col_score)
