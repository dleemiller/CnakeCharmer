"""Sliding window maximum using a monotonic deque.

Keywords: sliding window, maximum, monotonic deque, data structure, algorithms, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(100000,))
def deque_sliding_max(n: int) -> int:
    """Compute sliding window maximum over an array of n values with window size 1000.

    Uses a monotonic deque to maintain the window maximum in amortized O(1) per element.
    Returns the sum of all window maxima.

    Args:
        n: Array length.

    Returns:
        Sum of all sliding window maxima.
    """
    k = 1000
    # Generate array
    arr = [0] * n
    for i in range(n):
        arr[i] = ((i * 2654435761 + 13) ^ (i >> 3)) % 1000000

    # Monotonic deque (stores indices)
    deque = [0] * n
    head = 0
    tail = 0
    max_sum = 0

    for i in range(n):
        # Remove elements outside window
        while head < tail and deque[head] <= i - k:
            head += 1

        # Remove smaller elements from back
        while head < tail and arr[deque[tail - 1]] <= arr[i]:
            tail -= 1

        deque[tail] = i
        tail += 1

        if i >= k - 1:
            max_sum += arr[deque[head]]

    return max_sum
