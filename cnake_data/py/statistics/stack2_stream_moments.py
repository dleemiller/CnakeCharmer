"""Compute min/max/sum and second-moment stats in one pass.

Adapted from The Stack v2 Cython candidate:
- blob_id: 21db8f3c1be66b8f8adb0b5a5cd65816a0442406
- filename: statsfunctionscython.pyx

Keywords: statistics, streaming, moments, min max, variance numerator
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(850000, 29, 7))
def stack2_stream_moments(sample_count: int, shift_tag: int, stride_tag: int) -> tuple:
    """Generate integer signal and return streaming moment summary."""
    state = (987654321 + shift_tag * 2713) & 0xFFFFFFFF
    first = ((state >> 9) % 2001) - 1000

    min_val = first
    max_val = first
    total = 0
    sum_sq = 0

    for _step in range(sample_count):
        state = (1664525 * state + 1013904223 + stride_tag) & 0xFFFFFFFF
        val = ((state >> 9) % 2001) - 1000
        if val < min_val:
            min_val = val
        if val > max_val:
            max_val = val
        total += val
        sum_sq += val * val

    var_num = sample_count * sum_sq - total * total
    return (min_val, max_val, total & 0xFFFFFFFF, sum_sq & 0xFFFFFFFF, var_num & 0xFFFFFFFF)
