# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Otsu's threshold on a grayscale image (Cython-optimized).

Keywords: image processing, otsu, threshold, histogram, segmentation, cython, benchmark
"""

from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(500,))
def threshold(int n):
    """Compute optimal threshold via Otsu's method on n x n grayscale image."""
    cdef int hist[256]
    cdef int i, j, t, pixel, total, weight_bg, weight_fg
    cdef double sum_all, sum_bg, mean_bg, mean_fg, var_between, best_var
    cdef int best_thresh

    # Zero histogram
    for i in range(256):
        hist[i] = 0

    # Build histogram
    for i in range(n):
        for j in range(n):
            pixel = (i * 17 + j * 31 + 5) % 256
            hist[pixel] += 1

    total = n * n
    sum_all = 0.0
    for i in range(256):
        sum_all += i * hist[i]

    best_thresh = 0
    best_var = 0.0
    sum_bg = 0.0
    weight_bg = 0

    for t in range(256):
        weight_bg += hist[t]
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break

        sum_bg += t * hist[t]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_all - sum_bg) / weight_fg

        var_between = (<double>weight_bg) * (<double>weight_fg) * (mean_bg - mean_fg) * (mean_bg - mean_fg)

        if var_between > best_var:
            best_var = var_between
            best_thresh = t

    return best_thresh
