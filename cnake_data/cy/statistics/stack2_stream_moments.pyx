# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Compute min/max/sum and second-moment stats in one pass (Cython).

Adapted from The Stack v2 Cython candidate:
- blob_id: 21db8f3c1be66b8f8adb0b5a5cd65816a0442406
- filename: statsfunctionscython.pyx
"""

from cnake_data.benchmarks import cython_benchmark


cdef void _moments_kernel(
    int sample_count,
    int shift_tag,
    int stride_tag,
    int* min_out,
    int* max_out,
    unsigned int* total_out,
    unsigned int* sumsq_out,
    unsigned int* var_out,
) noexcept nogil:
    cdef unsigned int state = <unsigned int>(987654321 + shift_tag * 2713)
    cdef int step, val
    cdef int min_val = <int>((state >> 9) % 2001) - 1000
    cdef int max_val = min_val
    cdef long long total = 0
    cdef long long sum_sq = 0
    cdef long long var_num

    for step in range(sample_count):
        state = 1664525 * state + 1013904223 + <unsigned int>stride_tag
        val = <int>((state >> 9) % 2001) - 1000
        if val < min_val:
            min_val = val
        if val > max_val:
            max_val = val
        total += val
        sum_sq += val * val

    var_num = sample_count * sum_sq - total * total
    min_out[0] = min_val
    max_out[0] = max_val
    total_out[0] = <unsigned int>total
    sumsq_out[0] = <unsigned int>sum_sq
    var_out[0] = <unsigned int>var_num


@cython_benchmark(syntax="cy", args=(850000, 29, 7))
def stack2_stream_moments(int sample_count, int shift_tag, int stride_tag):
    cdef int min_val = 0
    cdef int max_val = 0
    cdef unsigned int total = 0
    cdef unsigned int sum_sq = 0
    cdef unsigned int var_num = 0
    with nogil:
        _moments_kernel(
            sample_count, shift_tag, stride_tag, &min_val, &max_val, &total, &sum_sq, &var_num
        )
    return (min_val, max_val, total, sum_sq, var_num)
