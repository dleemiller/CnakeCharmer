# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Cooley-Tukey radix-2 FFT of n complex values (Cython-optimized).

Keywords: dsp, fft, cooley-tukey, radix-2, fourier, complex, cython, benchmark
"""

from libc.math cimport cos, sin, sqrt, M_PI
from libc.stdlib cimport malloc, free
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(16384,))
def fft_radix2(int n):
    """Iterative Cooley-Tukey FFT using C arrays and nogil butterfly loop.

    Returns:
        Tuple of (mag_sum, mag_at_quarter, real_at_eighth).
    """
    # Largest power of 2 <= n
    cdef int size = 1
    while size * 2 <= n:
        size *= 2

    cdef double *x_r = <double *>malloc(size * sizeof(double))
    cdef double *x_i = <double *>malloc(size * sizeof(double))
    if x_r == NULL or x_i == NULL:
        if x_r != NULL: free(x_r)
        if x_i != NULL: free(x_i)
        raise MemoryError()

    cdef int log2_size = 0
    cdef int tmp = size
    while tmp > 1:
        log2_size += 1
        tmp >>= 1

    cdef int i, j, k, half, length, start
    cdef double TWO_PI = 2.0 * M_PI
    cdef double angle, wr, wi, cr, ci, ur, ui, vr, vi
    cdef double mag_sum, mag_at_quarter, real_at_eighth

    with nogil:
        # Input: pure complex tone at frequency 1
        for k in range(size):
            x_r[k] = cos(TWO_PI * k / size)
            x_i[k] = sin(TWO_PI * k / size)

        # Bit-reversal permutation
        for i in range(size):
            j = 0
            tmp = i
            for k in range(log2_size):
                j = (j << 1) | (tmp & 1)
                tmp >>= 1
            if j > i:
                ur = x_r[i]; x_r[i] = x_r[j]; x_r[j] = ur
                ui = x_i[i]; x_i[i] = x_i[j]; x_i[j] = ui

        # Butterfly stages — use while/C-style loops to avoid Python range objects
        length = 2
        while length <= size:
            half = length >> 1
            angle = -TWO_PI / length
            wr = cos(angle)
            wi = sin(angle)
            start = 0
            while start < size:
                cr = 1.0
                ci = 0.0
                for j in range(half):
                    ur = x_r[start + j]
                    ui = x_i[start + j]
                    vr = x_r[start + j + half] * cr - x_i[start + j + half] * ci
                    vi = x_r[start + j + half] * ci + x_i[start + j + half] * cr
                    x_r[start + j] = ur + vr
                    x_i[start + j] = ui + vi
                    x_r[start + j + half] = ur - vr
                    x_i[start + j + half] = ui - vi
                    cr, ci = cr * wr - ci * wi, cr * wi + ci * wr
                start += length
            length <<= 1

        # Summary statistics
        mag_sum = 0.0
        for k in range(size):
            mag_sum += sqrt(x_r[k] * x_r[k] + x_i[k] * x_i[k])
        k = size >> 2
        mag_at_quarter = sqrt(x_r[k] * x_r[k] + x_i[k] * x_i[k])
        k = size >> 3
        real_at_eighth = x_r[k]

    free(x_r)
    free(x_i)
    return (mag_sum, mag_at_quarter, real_at_eighth)
