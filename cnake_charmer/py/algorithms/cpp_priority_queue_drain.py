"""Max-heap with periodic draining using Python's heapq module.

Keywords: algorithms, heap, priority queue, max-heap, heapq, drain, benchmark
"""

import heapq

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(500000,))
def cpp_priority_queue_drain(n: int) -> tuple:
    """Push n values into a max-heap; drain top 10 every 100 pushes.

    Values pushed: (i * 2654435761) % 1000000
    Every 100 pushes, drain up to 10 maximum elements and accumulate their sum.

    Args:
        n: Total number of values to push.

    Returns:
        Tuple of (total_drained_sum, drain_count) where drain_count is the
        number of drain events performed.
    """
    heap: list[int] = []
    total_drained_sum = 0
    drain_count = 0

    for i in range(n):
        val = (i * 2654435761) % 1000000
        heapq.heappush(heap, -val)  # negate for max-heap via min-heap
        if (i + 1) % 100 == 0:
            for _ in range(10):
                if heap:
                    total_drained_sum += -heapq.heappop(heap)
            drain_count += 1

    return (total_drained_sum, drain_count)
