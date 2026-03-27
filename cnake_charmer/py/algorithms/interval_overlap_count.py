"""Count overlapping interval pairs using comparison-based sorting.

Keywords: algorithms, intervals, overlap, comparison, hash, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


class Interval:
    """Interval with start and end, comparable by start."""

    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __lt__(self, other):
        if self.start != other.start:
            return self.start < other.start
        return self.end < other.end

    def __le__(self, other):
        if self.start != other.start:
            return self.start < other.start
        return self.end <= other.end

    def __eq__(self, other):
        return self.start == other.start and self.end == other.end

    def __ne__(self, other):
        return self.start != other.start or self.end != other.end

    def __gt__(self, other):
        if self.start != other.start:
            return self.start > other.start
        return self.end > other.end

    def __ge__(self, other):
        if self.start != other.start:
            return self.start > other.start
        return self.end >= other.end

    def __hash__(self):
        return self.start * 100003 + self.end


@python_benchmark(args=(20000,))
def interval_overlap_count(n: int) -> int:
    """Create n intervals, sort by start, count overlapping pairs via sweep.

    Two intervals overlap if one starts before the other ends.

    Args:
        n: Number of intervals.

    Returns:
        Count of overlapping pairs.
    """
    intervals = []
    for i in range(n):
        start = ((i * 2654435761 + 17) ^ (i * 1103515245)) % 10000
        length = ((i * 1664525 + 1013904223) ^ (i * 214013)) % 500 + 1
        intervals.append(Interval(start, start + length))

    intervals.sort()

    # Sweep line: count pairs where interval[j].start < interval[i].end
    overlap_count = 0
    # Use a simple approach: for each interval, count how many previous
    # intervals have end > current start
    active_ends = []
    for i in range(n):
        # Remove ends that are <= current start
        new_active = []
        for e in active_ends:
            if e > intervals[i].start:
                new_active.append(e)
        active_ends = new_active
        overlap_count += len(active_ends)
        active_ends.append(intervals[i].end)

    return overlap_count
