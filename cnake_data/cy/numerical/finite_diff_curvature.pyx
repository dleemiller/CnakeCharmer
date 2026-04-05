# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Summarize second-difference curvature over a deterministic series (Cython)."""

from cnake_data.benchmarks import cython_benchmark


cdef void _curvature_kernel(
    int n,
    long long* sum_abs_out,
    int* max_abs_out,
    long long* signed_sum_out,
) noexcept nogil:
    cdef int i
    cdef int y0, y1, yi, yi2, d2, d2b, ad2
    cdef long long sum_abs = 0
    cdef int max_abs = 0
    cdef long long signed_sum = 0

    y0 = ((0 * 0 * 17 + 0 * 31 + 7) % 1000) - 500
    y1 = ((1 * 1 * 17 + 1 * 31 + 7) % 1000) - 500

    i = 2
    while i + 1 < n:
        yi = ((i * i * 17 + i * 31 + 7) % 1000) - 500
        d2 = yi - 2 * y1 + y0
        ad2 = d2 if d2 >= 0 else -d2
        sum_abs += ad2
        if ad2 > max_abs:
            max_abs = ad2
        signed_sum += d2 * ((i % 7) - 3)
        y0 = y1
        y1 = yi
        i += 1

        yi2 = ((i * i * 17 + i * 31 + 7) % 1000) - 500
        d2b = yi2 - 2 * y1 + y0
        ad2 = d2b if d2b >= 0 else -d2b
        sum_abs += ad2
        if ad2 > max_abs:
            max_abs = ad2
        signed_sum += d2b * ((i % 7) - 3)
        y0 = y1
        y1 = yi2
        i += 1

    while i < n:
        yi = ((i * i * 17 + i * 31 + 7) % 1000) - 500
        d2 = yi - 2 * y1 + y0
        ad2 = d2 if d2 >= 0 else -d2
        sum_abs += ad2
        if ad2 > max_abs:
            max_abs = ad2
        signed_sum += d2 * ((i % 7) - 3)
        y0 = y1
        y1 = yi
        i += 1

    sum_abs_out[0] = sum_abs
    max_abs_out[0] = max_abs
    signed_sum_out[0] = signed_sum


@cython_benchmark(syntax="cy", args=(200000,))
def finite_diff_curvature(int n):
    cdef long long sum_abs = 0
    cdef int max_abs = 0
    cdef long long signed_sum = 0

    if n < 3:
        return (0, 0, 0)

    with nogil:
        _curvature_kernel(n, &sum_abs, &max_abs, &signed_sum)

    return (sum_abs, max_abs, signed_sum)
