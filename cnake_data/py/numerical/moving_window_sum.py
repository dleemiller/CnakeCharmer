"""
Compute sliding window sum over a deterministically generated sequence.

Keywords: numerical, sliding window, moving sum, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(500000,))
def moving_window_sum(n: int) -> list:
    """Compute the sum of a sliding window of size 100 over a generated sequence.

    Generates sequence as value[i] = (i * 13 + 7) % 1000 for determinism,
    then computes the sum of each window of 100 consecutive elements.

    Args:
        n: Length of the sequence.

    Returns:
        List of ints of length n - 99, where each element is the sum of
        100 consecutive values starting at that index.
    """
    window = 100
    if n < window:
        return []

    values = [(i * 13 + 7) % 1000 for i in range(n)]

    current_sum = 0
    for i in range(window):
        current_sum += values[i]

    result = [current_sum]
    for i in range(window, n):
        current_sum += values[i] - values[i - window]
        result.append(current_sum)

    return result
