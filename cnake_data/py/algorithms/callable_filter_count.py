"""Count values passing a threshold filter using callable objects.

Keywords: algorithms, callable, filter, threshold, benchmark
"""

from cnake_data.benchmarks import python_benchmark


class ThresholdFilter:
    """Callable filter that returns True if value is within [lo, hi]."""

    def __init__(self, lo, hi):
        self.lo = lo
        self.hi = hi

    def __call__(self, x):
        return self.lo <= x <= self.hi


@python_benchmark(args=(500000,))
def callable_filter_count(n: int) -> int:
    """Apply threshold filters to n values, count how many pass.

    Args:
        n: Number of values to filter.

    Returns:
        Total count of values passing their assigned filter.
    """
    # Create 8 filters with different thresholds
    filters = []
    for i in range(8):
        lo = (i * 2654435761 + 17) % 500
        hi = lo + ((i * 1103515245 + 12345) % 500) + 1
        filters.append(ThresholdFilter(lo, hi))

    count = 0
    for i in range(n):
        val = ((i * 1664525 + 1013904223) ^ (i * 214013 + 2531011)) % 1000
        f = filters[i & 7]
        if f(val):
            count += 1

    return count
