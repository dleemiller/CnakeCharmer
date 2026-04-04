# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Transposed virtual error for synthetic biclusters (Cython)."""

from libc.stdlib cimport malloc, free
from libc.math cimport sqrt, fabs
from cnake_data.benchmarks import cython_benchmark

cdef unsigned int MASK32 = 0xFFFFFFFF


@cython_benchmark(syntax="cy", args=(512, 256, 17))
def bicluster_vet_score(int n_rows, int n_cols, int seed):
    cdef double *mat = <double *>malloc(n_rows * n_cols * sizeof(double))
    cdef double *row_means = <double *>malloc(n_rows * sizeof(double))
    cdef double *col_means = <double *>malloc(n_cols * sizeof(double))
    cdef double *col_stds = <double *>malloc(n_cols * sizeof(double))
    cdef unsigned int state
    cdef int i, j
    cdef double v, s1, s2, mu, var, mu_rho, sigma_rho2, sigma_rho, rr, cc, d, score

    if mat == NULL or row_means == NULL or col_means == NULL or col_stds == NULL:
        free(mat); free(row_means); free(col_means); free(col_stds)
        raise MemoryError()

    state = <unsigned int>((seed * 1664525 + 1013904223) & MASK32)
    for i in range(n_rows):
        row_means[i] = 0.0
        for j in range(n_cols):
            state = (state * 1664525 + 1013904223) & MASK32
            v = ((<int>(state % 2001)) - 1000) / 97.0
            mat[i * n_cols + j] = v + 0.01 * i - 0.02 * j

    for j in range(n_cols):
        s1 = 0.0
        s2 = 0.0
        for i in range(n_rows):
            v = mat[i * n_cols + j]
            s1 += v
            s2 += v * v
            row_means[i] += v
        mu = s1 / n_rows
        var = s2 / n_rows - mu * mu
        if var < 1e-12:
            var = 1e-12
        col_means[j] = mu
        col_stds[j] = sqrt(var)

    s1 = 0.0
    s2 = 0.0
    for i in range(n_rows):
        row_means[i] /= n_rows
        s1 += row_means[i]
        s2 += row_means[i] * row_means[i]
    mu_rho = s1 / n_rows
    sigma_rho2 = s2 / n_rows - mu_rho * mu_rho
    if sigma_rho2 < 1e-12:
        sigma_rho2 = 1e-12
    sigma_rho = sqrt(sigma_rho2)

    score = 0.0
    for i in range(n_rows):
        rr = (row_means[i] - mu_rho) / sigma_rho
        for j in range(n_cols):
            cc = (mat[i * n_cols + j] - col_means[j]) / col_stds[j]
            d = cc - rr
            score += fabs(d)
    score /= (n_rows * n_cols)

    free(mat); free(row_means); free(col_means); free(col_stds)
    return (score, mu_rho, sigma_rho)
