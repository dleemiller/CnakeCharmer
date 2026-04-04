# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False, initializedcheck=False, infer_types=True, language_level=3
"""Estimate conditional event probability with a lag window (Cython).

Sourced from SFT DuckDB blob: 109d9d93806ed7cd954b046b09e9ce7bab68c8b1
Keywords: conditional probability, event stream, lagged dependency, algorithms, cython
"""

from libc.stdlib cimport free, malloc

from cnake_data.benchmarks import cython_benchmark


cdef inline double _round_to(double x, int decimals):
    cdef double scale = 1.0
    cdef int i
    cdef double y
    for i in range(decimals):
        scale *= 10.0
    if x >= 0.0:
        y = x * scale + 0.5
    else:
        y = x * scale - 0.5
    return <long long>y / scale


cdef void _build_masks(int *masks, int rows, int window):
    cdef int i, j
    for i in range(rows):
        masks[i] = 0
        for j in range(window):
            if ((i + 3) * (j + 5) + 11) % 7 < 3:
                masks[i] |= 1 << j


cdef int _count_event(int *masks, int rows, int bit):
    cdef int i, cnt = 0
    for i in range(rows):
        if masks[i] & bit:
            cnt += 1
    return cnt


cdef int _count_lag_pair(int *masks, int rows, int delta, int prev_bit, int curr_bit):
    cdef int i, cnt = 0
    for i in range(delta, rows):
        if (masks[i - delta] & prev_bit) and (masks[i] & curr_bit):
            cnt += 1
    return cnt


@cython_benchmark(syntax="cy", args=(6, 100000, 2))
def conditional_event_prob(int window, int rows, int delta):
    cdef int bit
    cdef int a_event
    cdef int p_a = 0
    cdef int p_b = 0
    cdef int p_ba = 0
    cdef int denom
    cdef double p_a_f, p_b_f, p_ba_f
    cdef int *masks = NULL

    if delta < 0:
        raise ValueError("delta must be non-negative")
    if rows <= 0 or window <= 0:
        return (0.0, 0.0, 0.0)
    if window > 30:
        raise ValueError("window must be <= 30")

    masks = <int *>malloc(rows * sizeof(int))
    if masks == NULL:
        raise MemoryError()

    _build_masks(masks, rows, window)

    a_event = window - 1
    bit = 1 << a_event

    p_a = _count_event(masks, rows, bit)
    p_b = _count_event(masks, rows, 1)
    p_ba = _count_lag_pair(masks, rows, delta, 1, bit)
    free(masks)

    p_a_f = p_a / <double>rows if rows > 0 else 0.0
    p_b_f = p_b / <double>rows if rows > 0 else 0.0
    denom = rows - delta
    if denom < 1:
        denom = 1
    p_ba_f = p_ba / <double>denom
    return (_round_to(p_ba_f, 10), _round_to(p_a_f, 10), _round_to(p_b_f, 10))
