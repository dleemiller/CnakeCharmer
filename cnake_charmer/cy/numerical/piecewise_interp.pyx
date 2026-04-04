# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""
Piecewise linear interpolation (AFGEN algorithm) (Cython-optimized).

Keywords: numerical, interpolation, piecewise, linear, afgen, cython, benchmark
"""

from libc.math cimport sin, M_PI
from libc.stdlib cimport malloc, free
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(5000,))
def piecewise_interp(int n):
    """Evaluate piecewise linear interpolation using C arrays and nogil."""
    cdef int i, j
    cdef double t, x0, x1, y0, y1, alpha, sum_results
    cdef double two_pi = 2.0 * M_PI

    cdef double *x = <double *>malloc(n * sizeof(double))
    cdef double *y = <double *>malloc(n * sizeof(double))
    cdef double *results = <double *>malloc(n * sizeof(double))

    if not x or not y or not results:
        if x: free(x)
        if y: free(y)
        if results: free(results)
        raise MemoryError("Failed to allocate arrays")

    with nogil:
        # Fill x and y arrays
        for i in range(n):
            x[i] = <double>i
            y[i] = sin(two_pi * i / n) * 100.0

        # Evaluate at m=n query points
        for j in range(n):
            t = j * (n - 1) / (<double>n)

            if t <= x[0]:
                results[j] = y[0]
            elif t >= x[n - 1]:
                results[j] = y[n - 1]
            else:
                # x[i]=i so bracket is at i = floor(t)+1
                i = <int>t + 1
                if i >= n:
                    i = n - 1
                x0 = x[i - 1]
                x1 = x[i]
                y0 = y[i - 1]
                y1 = y[i]
                alpha = (t - x0) / (x1 - x0)
                results[j] = y0 + alpha * (y1 - y0)

        sum_results = 0.0
        for j in range(n):
            sum_results += results[j]

    cdef long long r0 = <long long>(sum_results * 1000) % (10 ** 9)
    cdef long long r1 = <long long>(results[n // 3] * 1e6)

    free(x)
    free(y)
    free(results)
    return (r0, r1)
