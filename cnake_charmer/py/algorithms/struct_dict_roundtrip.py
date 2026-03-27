"""Compute statistics for groups using struct-dict pattern.

Demonstrates struct-to-dict roundtrip: compute Stats per
group and return as list of dicts.

Keywords: algorithms, struct, dict, roundtrip, statistics, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(10000,))
def struct_dict_roundtrip(n: int) -> float:
    """Compute stats for n groups, return total of means.

    Each group has 10 elements derived from hash. Computes
    mean, std, count for each group.

    Args:
        n: Number of groups.

    Returns:
        Sum of all group means.
    """
    mask = 0xFFFFFFFF
    group_size = 10
    total_mean = 0.0

    for g in range(n):
        s = 0.0
        s2 = 0.0
        for k in range(group_size):
            idx = g * group_size + k
            h = ((idx * 2654435761) & mask) ^ ((idx * 2246822519) & mask)
            val = (h & 0xFFFF) / 65535.0
            s += val
            s2 += val * val

        mean = s / group_size
        variance = s2 / group_size - mean * mean
        # Create dict (struct auto-conversion analog)
        stats = {"mean": mean, "std": variance**0.5, "count": group_size}
        total_mean += stats["mean"]

    return total_mean
