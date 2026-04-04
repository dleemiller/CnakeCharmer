# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Thomas algorithm solve statistics for generated tridiagonal systems (Cython)."""

from libc.stdlib cimport malloc, free

from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(0.11, 2.2, 0.13, 1.7, 800, 3))
def tdma_solver_stats(double a0, double b0, double c0, double d0, int n, int passes):
    cdef double *a = <double *>malloc(n * sizeof(double))
    cdef double *b = <double *>malloc(n * sizeof(double))
    cdef double *c = <double *>malloc(n * sizeof(double))
    cdef double *d = <double *>malloc(n * sizeof(double))
    cdef double *x = <double *>malloc(n * sizeof(double))
    cdef int i, p
    cdef double m, a_i, b_i, c_i, d_i, r
    cdef double checksum = 0.0
    cdef double residual_sum = 0.0
    cdef double last_x0 = 0.0

    if not a or not b or not c or not d or not x:
        free(a)
        free(b)
        free(c)
        free(d)
        free(x)
        raise MemoryError()

    for p in range(passes):
        for i in range(n):
            a[i] = a0 + 0.0003 * ((i + p) % 11)
            b[i] = b0 + 0.0002 * ((i + 2 * p) % 13)
            c[i] = c0 + 0.00025 * ((i + 3 * p) % 7)
            d[i] = d0 + 0.001 * ((i + 5 * p) % 17)

        for i in range(1, n):
            m = a[i - 1] / b[i - 1]
            b[i] -= m * c[i - 1]
            d[i] -= m * d[i - 1]

        x[n - 1] = d[n - 1] / b[n - 1]
        for i in range(n - 2, -1, -1):
            x[i] = (d[i] - c[i] * x[i + 1]) / b[i]

        for i in range(1, n - 1):
            a_i = a0 + 0.0003 * ((i + p) % 11)
            b_i = b0 + 0.0002 * ((i + 2 * p) % 13)
            c_i = c0 + 0.00025 * ((i + 3 * p) % 7)
            d_i = d0 + 0.001 * ((i + 5 * p) % 17)
            r = a_i * x[i - 1] + b_i * x[i] + c_i * x[i + 1] - d_i
            if r >= 0.0:
                residual_sum += r
            else:
                residual_sum -= r

        checksum += x[0] + x[n // 2] + x[n - 1]
        last_x0 = x[0]

    free(a)
    free(b)
    free(c)
    free(d)
    free(x)
    return (checksum, residual_sum, last_x0)
