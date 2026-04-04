"""Group-by-key aggregation using defaultdict.

Keywords: group by, aggregation, hash map, defaultdict, benchmark
"""

from collections import defaultdict

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(1000000,))
def stl_unordered_map_group_sum(n: int) -> tuple:
    """Aggregate values into groups and return discriminating statistics.

    Keys and values are generated deterministically:
        num_groups = n // 10
        key_i   = (i * 2654435761) % num_groups
        value_i = (i * 1103515245 + 12345) % 1000

    Args:
        n: Number of (key, value) pairs to process.

    Returns:
        Tuple of (xor_of_all_group_sums, max_group_sum).
    """
    num_groups = n // 10
    groups: dict[int, int] = defaultdict(int)

    for i in range(n):
        key = (i * 2654435761) % num_groups
        value = (i * 1103515245 + 12345) % 1000
        groups[key] += value

    xor_sum = 0
    max_sum = 0
    for s in groups.values():
        xor_sum ^= s
        if s > max_sum:
            max_sum = s

    return (xor_sum, max_sum)
