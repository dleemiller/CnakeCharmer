"""Compute Kolmogorov-Smirnov statistic between two deterministic distributions.

Keywords: statistics, kolmogorov-smirnov, ks-test, ecdf, distribution, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(200000,))
def kolmogorov_smirnov(n: int) -> tuple:
    """Compute KS statistic between two deterministic sample distributions.

    Sample A: a[i] = (i*73+11) % 4999, size n.
    Sample B: b[i] = (i*97+13) % 5003, size n.
    Merges sorted samples and walks ECDFs to find max difference.

    Args:
        n: Size of each sample.

    Returns:
        Tuple of (ks_stat, max_diff_position, ecdf_mid).
    """
    a = [(i * 73 + 11) % 4999 for i in range(n)]
    b = [(i * 97 + 13) % 5003 for i in range(n)]

    a.sort()
    b.sort()

    ks_stat = 0.0
    max_diff_pos = 0
    i = 0
    j = 0
    inv_n = 1.0 / n

    while i < n and j < n:
        if a[i] <= b[j]:
            i += 1
        else:
            j += 1

        ecdf_a = i * inv_n
        ecdf_b = j * inv_n
        diff = ecdf_a - ecdf_b
        if diff < 0:
            diff = -diff
        if diff > ks_stat:
            ks_stat = diff
            max_diff_pos = i + j

    # Handle remaining elements
    while i < n:
        i += 1
        ecdf_a = i * inv_n
        ecdf_b = j * inv_n
        diff = ecdf_a - ecdf_b
        if diff < 0:
            diff = -diff
        if diff > ks_stat:
            ks_stat = diff
            max_diff_pos = i + j

    while j < n:
        j += 1
        ecdf_a = i * inv_n
        ecdf_b = j * inv_n
        diff = ecdf_a - ecdf_b
        if diff < 0:
            diff = -diff
        if diff > ks_stat:
            ks_stat = diff
            max_diff_pos = i + j

    # ecdf_mid: ECDF_A value at the midpoint index of sample A
    mid_idx = n // 2
    ecdf_mid = (mid_idx + 1) * inv_n

    return (ks_stat, max_diff_pos, ecdf_mid)
