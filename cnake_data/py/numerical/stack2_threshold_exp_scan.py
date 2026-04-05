"""Thresholded exponential scan across generated scalar series.

Adapted from The Stack v2 Cython candidate:
- blob_id: c5a83df9c1b6b624fa118d7e7eb14b6acd9ea353
- filename: dot_product.pyx

Keywords: numerical, threshold, exponential, scan, accumulation
"""

from math import exp

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(2200000, 620, 13))
def stack2_threshold_exp_scan(vector_size: int, threshold_milli: int, seed_tag: int) -> tuple:
    """Accumulate transformed values above threshold and summarize outputs."""
    state = (987123 + seed_tag * 4129) & 0x7FFFFFFF
    threshold = threshold_milli / 1000.0

    active = 0
    total_scaled = 0
    checksum = 0
    last_scaled = 0

    for idx in range(vector_size):
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        val = (state & 0xFFFF) / 65535.0
        if val > threshold:
            scaled = int(exp(val) * 1000.0)
            active += 1
            total_scaled += scaled
            last_scaled = scaled
            checksum = (checksum + scaled * (idx + 1)) & 0xFFFFFFFF

    return (active, total_scaled & 0xFFFFFFFF, checksum, last_scaled)
