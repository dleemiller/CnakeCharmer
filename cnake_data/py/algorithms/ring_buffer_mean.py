"""Ring buffer running mean computation.

Keywords: ring buffer, circular buffer, running mean, data structure, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(100000,))
def ring_buffer_mean(n: int) -> float:
    """Push n values into a fixed-size ring buffer and accumulate running means.

    Each value is pushed into a ring buffer of capacity 1000. After each push,
    the mean of the buffer contents is added to a running sum.

    Args:
        n: Number of values to push.

    Returns:
        Sum of all running means.
    """
    capacity = 1000
    buf = [0.0] * capacity
    head = 0
    size = 0
    total = 0.0
    mean_sum = 0.0

    for i in range(n):
        val = (i * 7 + 13) % 10007 / 100.0
        if size == capacity:
            total -= buf[head]
        else:
            size += 1
        buf[head] = val
        total += val
        head = (head + 1) % capacity
        mean_sum += total / size

    return mean_sum
