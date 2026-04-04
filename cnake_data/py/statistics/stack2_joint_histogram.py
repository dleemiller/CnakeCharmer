"""Build a joint histogram for deterministic paired signals.

Adapted from The Stack v2 Cython candidate:
- blob_id: b67354d664bd9d79078d1fc8f8bb0e617479e227
- filename: histogram_2D.pyx

Keywords: statistics, histogram, joint distribution, bins, matrix
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(260000, 48, 23))
def stack2_joint_histogram(sample_count: int, bin_count: int, scale_tag: int) -> tuple:
    """Accumulate a 2D histogram from generated value pairs."""
    state = (1357911 + scale_tag * 4099) & 0xFFFFFFFF
    hist = [[0] * bin_count for _ in range(bin_count)]

    for _ in range(sample_count):
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        left = (state >> 4) & 0xFFFF
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        right = (state >> 7) & 0xFFFF
        bx = (left * bin_count) >> 16
        by = (right * bin_count) >> 16
        hist[bx][by] += 1

    max_bin = 0
    diag_sum = 0
    checksum = 0
    for row in range(bin_count):
        for col in range(bin_count):
            val = hist[row][col]
            if val > max_bin:
                max_bin = val
            if row == col:
                diag_sum += val
            checksum = (checksum + val * (row + 1) * (col + 3)) & 0xFFFFFFFF

    return (max_bin, diag_sum, checksum, hist[0][bin_count - 1])
