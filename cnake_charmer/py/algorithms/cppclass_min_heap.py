"""Min-heap priority queue benchmark using heapq.

Keywords: heap, priority queue, min-heap, heapq, benchmark
"""

import heapq

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(500000,))
def cppclass_min_heap(n: int) -> tuple:
    """Process n heap operations: push then alternating pop/push.

    First 3n/4 operations are pushes of (i * 2654435761) % 1_000_000.
    Remaining n/4 operations alternate pop_min and push of a new value.

    Args:
        n: Total number of operations.

    Returns:
        Tuple of (sum_of_popped_values, pop_count).
    """
    heap: list[int] = []
    push_count = (n * 3) // 4
    remaining = n - push_count

    for i in range(push_count):
        val = (i * 2654435761) % 1_000_000
        heapq.heappush(heap, val)

    sum_popped = 0
    pop_count = 0
    for j in range(remaining):
        if j % 2 == 0:
            # pop
            if heap:
                sum_popped += heapq.heappop(heap)
                pop_count += 1
        else:
            # push
            val = ((push_count + j) * 2654435761) % 1_000_000
            heapq.heappush(heap, val)

    return (sum_popped, pop_count)
