"""Count synthetic line segments that cross the dateline.

Keywords: geometry, dateline, longitude, crossing, benchmark
"""

from cnake_data.benchmarks import python_benchmark


def _crosses(x0: float, x1: float) -> bool:
    if x0 == x1:
        return False
    if not ((x0 < 0.0 < x1) or (x1 < 0.0 < x0)):
        return False
    return abs(x1 - x0) > 180.0


@python_benchmark(args=(7, 200000))
def crosses_dateline_count(seed: int, segment_count: int) -> tuple[int, int]:
    """Count crossings with an extra checksum to discriminate failures."""
    count = 0
    checksum = 0
    state = seed & 0xFFFFFFFF
    for _ in range(segment_count):
        state = (1103515245 * state + 12345) & 0xFFFFFFFF
        x0 = ((state >> 8) % 360) - 180.0
        state = (1103515245 * state + 12345) & 0xFFFFFFFF
        x1 = ((state >> 8) % 360) - 180.0
        if _crosses(x0, x1):
            count += 1
            checksum += int((x1 - x0) * 10.0)
    return count, checksum
