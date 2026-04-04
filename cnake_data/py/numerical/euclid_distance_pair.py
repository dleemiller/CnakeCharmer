"""Pairwise 2D Euclidean distance accumulation.

Keywords: numerical, euclidean distance, geometry, loop, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(17, 40000, 0.01))
def euclid_distance_pair(seed: int, pair_count: int, scale: float) -> float:
    """Accumulate synthetic pairwise 2D distances.

    Args:
        seed: Initial RNG seed.
        pair_count: Number of generated point pairs.
        scale: Coordinate scale factor.

    Returns:
        Sum of Euclidean distances.
    """
    total = 0.0
    state = seed & 0xFFFFFFFF
    for _ in range(pair_count):
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        x1 = ((state >> 8) & 0xFFFF) * scale
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        y1 = ((state >> 8) & 0xFFFF) * scale
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        x2 = ((state >> 8) & 0xFFFF) * scale
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        y2 = ((state >> 8) & 0xFFFF) * scale
        dx = x1 - x2
        dy = y1 - y2
        total += (dx * dx + dy * dy) ** 0.5
    return total
