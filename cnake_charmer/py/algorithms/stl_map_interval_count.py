"""Online frequency counting with sorted min/max tracking.

Maintains a frequency counter while tracking the current minimum
and maximum keys. Demonstrates a use case where sorted container
insert + extrema access matters.

Keywords: algorithms, sorted map, online, frequency, min max, benchmark
"""

import bisect

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(200000,))
def stl_map_interval_count(n: int) -> tuple:
    """Insert keys into a sorted frequency counter, tracking min/max spread.

    For each i, insert key = hash(i) % key_range. Every 8th step,
    also delete the oldest key (sliding window). Track sum of
    (max_key - min_key) across all steps where the container is non-empty.

    Python maintains a sorted list via bisect.insort (O(n) per insert).

    Args:
        n: Number of operations.

    Returns:
        Tuple of (final_distinct_count, spread_sum).
    """
    key_range = n * 4
    sorted_keys = []
    freq = {}
    spread_sum = 0
    window = []

    for i in range(n):
        key = (i * 2654435761) % key_range
        window.append(key)

        if key not in freq:
            freq[key] = 0
            bisect.insort(sorted_keys, key)
        freq[key] += 1

        # Sliding window: remove oldest every 8 steps
        if i >= 8 and i & 7 == 0:
            old_key = window[i - 8]
            freq[old_key] -= 1
            if freq[old_key] == 0:
                del freq[old_key]
                idx = bisect.bisect_left(sorted_keys, old_key)
                sorted_keys.pop(idx)

        if sorted_keys:
            spread_sum += sorted_keys[-1] - sorted_keys[0]

    return (len(freq), spread_sum)
