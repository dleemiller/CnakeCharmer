"""Find kth smallest element using a max-heap of size k.

Keywords: heap, kth smallest, selection, max-heap, sorting, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(100000,))
def heap_kth_smallest(n: int) -> int:
    """Stream n values through a max-heap of size k=100 to find kth smallest.

    Maintains a max-heap of the k smallest values seen so far.
    Returns the sum of the kth-smallest value after each of the last n-k insertions.

    Args:
        n: Number of values to stream.

    Returns:
        Sum of kth-smallest snapshots.
    """
    k = 100

    # Max-heap implemented with negation on a min-heap structure
    heap = []
    heap_size = 0
    result_sum = 0

    for i in range(n):
        val = ((i * 2654435761 + 17) ^ (i * 1103515245)) % 1000000

        if heap_size < k:
            # Insert into heap (max-heap via negation)
            heap.append(-val)
            heap_size += 1
            _sift_up(heap, heap_size - 1)
            if heap_size == k:
                result_sum += -heap[0]
        else:
            if val < -heap[0]:
                heap[0] = -val
                _sift_down(heap, 0, heap_size)
            result_sum += -heap[0]

    return result_sum


def _sift_up(heap, pos):
    while pos > 0:
        parent = (pos - 1) >> 1
        if heap[pos] < heap[parent]:
            heap[pos], heap[parent] = heap[parent], heap[pos]
            pos = parent
        else:
            break


def _sift_down(heap, pos, size):
    while True:
        left = 2 * pos + 1
        right = 2 * pos + 2
        smallest = pos
        if left < size and heap[left] < heap[smallest]:
            smallest = left
        if right < size and heap[right] < heap[smallest]:
            smallest = right
        if smallest != pos:
            heap[pos], heap[smallest] = heap[smallest], heap[pos]
            pos = smallest
        else:
            break
