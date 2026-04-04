"""Sum of medians of a sliding window over a deterministic sequence.

Keywords: statistics, moving median, sliding window, median, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(1000000,))
def moving_median(n: int) -> int:
    """Compute sum of medians of a sliding window of size 5.

    Sequence: v[i] = (i*13 + 7) % 1000
    For each window of 5 consecutive elements, find the median and sum them all.
    Uses insertion into a sorted buffer of 5 elements.

    Args:
        n: Length of the sequence.

    Returns:
        Sum of all window medians (as int).
    """
    if n < 5:
        return 0
    total = 0
    for i in range(n - 4):
        # Collect 5 elements
        buf = [0] * 5
        for j in range(5):
            buf[j] = ((i + j) * 13 + 7) % 1000
        # Insertion sort the 5-element buffer
        for j in range(1, 5):
            key = buf[j]
            k = j - 1
            while k >= 0 and buf[k] > key:
                buf[k + 1] = buf[k]
                k -= 1
            buf[k + 1] = key
        total += buf[2]
    return total
