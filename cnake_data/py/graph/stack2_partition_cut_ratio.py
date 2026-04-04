"""Compute partition cut and ratio metrics on deterministic weighted graphs.

Adapted from The Stack v2 Cython candidate:
- blob_id: 7035f318b0301a1b1758627b664e7ddba244321f
- filename: tools.pyx

Keywords: graph, partition, cut ratio, weighted edges, degree
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(620, 9, 29))
def stack2_partition_cut_ratio(node_count: int, group_mod: int, seed_tag: int) -> tuple:
    """Build weighted graph and summarize cross/internal edge structure."""
    if group_mod <= 0:
        return (0, 0, 0, 0)

    cross_sum = 0
    internal_sum = 0
    max_degree = 0
    degrees = [0] * node_count

    for left in range(node_count):
        left_group = left % group_mod
        for right in range(left + 1, node_count):
            weight = (left * 131 + right * 17 + seed_tag * 19) % 31
            if weight < 6:
                continue

            right_group = right % group_mod
            degrees[left] += weight
            degrees[right] += weight
            if left_group == right_group:
                internal_sum += weight
            else:
                cross_sum += weight

    for val in degrees:
        if val > max_degree:
            max_degree = val

    ratio_scaled = (cross_sum * 1_000_000) // (internal_sum + 1)
    checksum = (cross_sum * 97 + internal_sum * 53 + max_degree * 11) & 0xFFFFFFFF
    return (cross_sum, internal_sum, ratio_scaled, checksum)
