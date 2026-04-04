"""Sort priority items using comparison-based sorting with priority/value pairs.

Keywords: sorting, priority queue, comparison, richcmp, benchmark
"""

from cnake_data.benchmarks import python_benchmark


class PriorityItem:
    """Item with priority and value, comparable by priority then value."""

    def __init__(self, priority, value):
        self.priority = priority
        self.value = value

    def __lt__(self, other):
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.value < other.value

    def __le__(self, other):
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.value <= other.value

    def __eq__(self, other):
        return self.priority == other.priority and self.value == other.value

    def __ne__(self, other):
        return self.priority != other.priority or self.value != other.value

    def __gt__(self, other):
        if self.priority != other.priority:
            return self.priority > other.priority
        return self.value > other.value

    def __ge__(self, other):
        if self.priority != other.priority:
            return self.priority > other.priority
        return self.value >= other.value


@python_benchmark(args=(50000,))
def priority_queue_sort(n: int) -> int:
    """Build n PriorityItem objects, sort them, return checksum of sorted order.

    Args:
        n: Number of items to create and sort.

    Returns:
        Checksum of sorted item positions.
    """
    items = []
    for i in range(n):
        priority = ((i * 2654435761) ^ (i * 1103515245 + 12345)) % 1000
        value = ((i * 1664525 + 1013904223) ^ (i * 214013)) % 100000
        items.append(PriorityItem(priority, value))

    items.sort()

    checksum = 0
    for i in range(n):
        checksum = (
            checksum * 31 + items[i].priority * 1000000 + items[i].value
        ) & 0x7FFFFFFFFFFFFFFF
    return checksum
