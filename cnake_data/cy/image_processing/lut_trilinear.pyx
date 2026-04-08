# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""3D LUT with trilinear interpolation for color grading — Cython implementation."""

from libc.math cimport cos, sin
from libc.stdlib cimport free, malloc

from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(80, 80, 17))
def lut_trilinear(int rows, int cols, int lut_size):
    """Apply a 3D LUT with trilinear interpolation to a synthetic RGB image."""
    cdef int n = lut_size
    cdef int lut_n = n * n * n
    cdef double *lut_r = <double *>malloc(lut_n * sizeof(double))
    cdef double *lut_g = <double *>malloc(lut_n * sizeof(double))
    cdef double *lut_b = <double *>malloc(lut_n * sizeof(double))
    if not lut_r or not lut_g or not lut_b:
        free(lut_r); free(lut_g); free(lut_b)
        raise MemoryError()

    cdef double step = 1.0 / (n - 1)
    cdef double r_in, g_in, b_in, lum
    cdef int ri, gi, bi, idx

    for ri in range(n):
        r_in = ri * step
        for gi in range(n):
            g_in = gi * step
            for bi in range(n):
                b_in = bi * step
                idx = ri * n * n + gi * n + bi
                lum = 0.299 * r_in + 0.587 * g_in + 0.114 * b_in
                lut_r[idx] = r_in * 1.1 + lum * 0.05
                if lut_r[idx] > 1.0:
                    lut_r[idx] = 1.0
                lut_g[idx] = g_in * 0.97 + lum * 0.03
                if lut_g[idx] > 1.0:
                    lut_g[idx] = 1.0
                lut_b[idx] = b_in * 0.88 - lum * 0.02
                if lut_b[idx] < 0.0:
                    lut_b[idx] = 0.0

    cdef double sum_r_out = 0.0, sum_g_out = 0.0, sum_b_out = 0.0
    cdef double ri_f, gi_f, bi_f, fr, fg, fb, mfr, mfg, mfb
    cdef double w000, w100, w010, w110, w001, w101, w011, w111
    cdef int r0, g0, b0, r1, g1, b1, row, col

    for row in range(rows):
        for col in range(cols):
            r_in = sin(row * 0.05) * 0.5 + 0.5
            g_in = cos(col * 0.05) * 0.5 + 0.5
            b_in = sin((row + col) * 0.03) * 0.5 + 0.5

            ri_f = r_in * (n - 1)
            gi_f = g_in * (n - 1)
            bi_f = b_in * (n - 1)
            r0 = <int>ri_f
            g0 = <int>gi_f
            b0 = <int>bi_f
            if r0 >= n - 1:
                r0 = n - 2
            if g0 >= n - 1:
                g0 = n - 2
            if b0 >= n - 1:
                b0 = n - 2
            r1 = r0 + 1
            g1 = g0 + 1
            b1 = b0 + 1
            fr = ri_f - r0
            fg = gi_f - g0
            fb = bi_f - b0
            mfr = 1.0 - fr
            mfg = 1.0 - fg
            mfb = 1.0 - fb

            w000 = mfr * mfg * mfb
            w100 = fr  * mfg * mfb
            w010 = mfr * fg  * mfb
            w110 = fr  * fg  * mfb
            w001 = mfr * mfg * fb
            w101 = fr  * mfg * fb
            w011 = mfr * fg  * fb
            w111 = fr  * fg  * fb

            sum_r_out += (w000 * lut_r[r0*n*n + g0*n + b0]
                        + w100 * lut_r[r1*n*n + g0*n + b0]
                        + w010 * lut_r[r0*n*n + g1*n + b0]
                        + w110 * lut_r[r1*n*n + g1*n + b0]
                        + w001 * lut_r[r0*n*n + g0*n + b1]
                        + w101 * lut_r[r1*n*n + g0*n + b1]
                        + w011 * lut_r[r0*n*n + g1*n + b1]
                        + w111 * lut_r[r1*n*n + g1*n + b1])
            sum_g_out += (w000 * lut_g[r0*n*n + g0*n + b0]
                        + w100 * lut_g[r1*n*n + g0*n + b0]
                        + w010 * lut_g[r0*n*n + g1*n + b0]
                        + w110 * lut_g[r1*n*n + g1*n + b0]
                        + w001 * lut_g[r0*n*n + g0*n + b1]
                        + w101 * lut_g[r1*n*n + g0*n + b1]
                        + w011 * lut_g[r0*n*n + g1*n + b1]
                        + w111 * lut_g[r1*n*n + g1*n + b1])
            sum_b_out += (w000 * lut_b[r0*n*n + g0*n + b0]
                        + w100 * lut_b[r1*n*n + g0*n + b0]
                        + w010 * lut_b[r0*n*n + g1*n + b0]
                        + w110 * lut_b[r1*n*n + g1*n + b0]
                        + w001 * lut_b[r0*n*n + g0*n + b1]
                        + w101 * lut_b[r1*n*n + g0*n + b1]
                        + w011 * lut_b[r0*n*n + g1*n + b1]
                        + w111 * lut_b[r1*n*n + g1*n + b1])

    free(lut_r); free(lut_g); free(lut_b)
    return (sum_r_out, sum_g_out, sum_b_out)
