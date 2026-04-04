"""Weighted random-choice walk with deterministic LCG state.

Adapted from The Stack v2 Cython candidate:
- blob_id: 34eb7bf393034a3b474c6dffead875c6125c37fe
- filename: rng.pyx

Keywords: statistics, rng, weighted choice, histogram, checksum
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(32, 220000, 17))
def stack2_rng_weighted_walk(bucket_count: int, draw_count: int, seed_offset: int) -> tuple:
    """Sample weighted buckets and return compact summary stats."""
    state = (123456789 + seed_offset * 7919) & 0xFFFFFFFF
    weights = [0] * bucket_count

    total_weight = 0
    for idx in range(bucket_count):
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        val = ((state >> 8) % 1000) + 1
        weights[idx] = val
        total_weight += val

    counts = [0] * bucket_count
    checksum = 0
    top_idx = 0

    for step in range(draw_count):
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        target = state % total_weight
        partial = 0
        pick = 0
        for idx, wt in enumerate(weights):
            partial += wt
            if target < partial:
                pick = idx
                break

        counts[pick] += 1
        checksum = (checksum + (pick + 3) * (step + 11)) & 0xFFFFFFFF
        if counts[pick] > counts[top_idx]:
            top_idx = pick

    return (top_idx, counts[top_idx], counts[-1], checksum)
