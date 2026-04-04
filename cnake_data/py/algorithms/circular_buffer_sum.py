"""Accumulate windowed sums using a circular buffer.

Keywords: circular buffer, windowed sum, container, sequence protocol, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(100000,))
def circular_buffer_sum(n: int) -> float:
    """Push n values into a circular buffer, accumulate sum of last-k elements.

    At each step, reads back the last min(size, 100) elements via indexing
    and sums them. Returns the grand total of all per-step sums.

    Args:
        n: Number of values to push.

    Returns:
        Grand total of windowed sums.
    """
    capacity = 500
    buf = [0.0] * capacity
    head = 0
    size = 0
    grand_total = 0.0
    k = 100

    for i in range(n):
        val = ((i * 2654435761 + 7) % 10000) / 100.0

        # Write at head
        pos = head % capacity
        buf[pos] = val
        head += 1  # noqa: SIM113
        if size < capacity:
            size += 1

        # Sum last k elements via indexing
        window = min(size, k)
        window_sum = 0.0
        for j in range(window):
            # Index from most recent backwards
            idx = (head - 1 - j) % size
            # Adjust for circular buffer
            actual = idx % capacity
            window_sum += buf[actual]

        grand_total += window_sum

    return grand_total
