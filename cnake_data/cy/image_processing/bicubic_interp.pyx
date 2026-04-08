# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Bicubic interpolation for image upsampling — Cython implementation."""

from libc.math cimport cos, sin

from cnake_data.benchmarks import cython_benchmark


cdef double _cubic_weight(double t) nogil:
    """Keys cubic convolution kernel weight."""
    if t < 0.0:
        t = -t
    if t < 1.0:
        return 1.5 * t * t * t - 2.5 * t * t + 1.0
    if t < 2.0:
        return -0.5 * t * t * t + 2.5 * t * t - 4.0 * t + 2.0
    return 0.0


@cython_benchmark(syntax="cy", args=(80, 80, 2))
def bicubic_interp(int src_rows, int src_cols, int scale):
    """Bicubic upsample a deterministic image by integer scale factor."""
    cdef int dst_rows = src_rows * scale
    cdef int dst_cols = src_cols * scale
    cdef int cr = dst_rows // 2
    cdef int cc = dst_cols // 2

    cdef double total = 0.0
    cdef double center_val = 0.0
    cdef double corner_val = 0.0
    cdef double sy, sx, fy, fx, val, wy, wx, src_val
    cdef int dr, dc, m, n, iy, ix, sr, sc

    for dr in range(dst_rows):
        sy = <double>dr / scale
        iy = <int>sy
        fy = sy - iy
        for dc in range(dst_cols):
            sx = <double>dc / scale
            ix = <int>sx
            fx = sx - ix
            val = 0.0
            for m in range(-1, 3):
                wy = _cubic_weight(m - fy)
                sr = iy + m
                if sr < 0:
                    sr = 0
                elif sr >= src_rows:
                    sr = src_rows - 1
                for n in range(-1, 3):
                    wx = _cubic_weight(n - fx)
                    sc = ix + n
                    if sc < 0:
                        sc = 0
                    elif sc >= src_cols:
                        sc = src_cols - 1
                    src_val = sin(sr * 0.3) * cos(sc * 0.2)
                    val += wy * wx * src_val
            total += val
            if dr == cr and dc == cc:
                center_val = val
            if dr == 0 and dc == 0:
                corner_val = val

    return (total, center_val, corner_val)
