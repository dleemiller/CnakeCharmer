"""Euclidean distance between signal arrays.

Computes pairwise Euclidean distances between multiple signal pairs using
a manual Newton's method sqrt implementation.

Keywords: euclidean, distance, signal, sqrt, newton, numerical, pairwise
"""

from cnake_charmer.benchmarks import python_benchmark

N = 5000


@python_benchmark(args=(N,))
def signal_distance(n: int) -> tuple:
    """Compute pairwise Euclidean distances for n signal pairs of length 64.

    Args:
        n: Number of signal pairs.

    Returns:
        Tuple of (total_distance, max_distance).
    """
    sig_len = 64
    total_signals = 2 * n

    # Generate signals deterministically
    signals = []
    for i in range(total_signals):
        sig = []
        for j in range(sig_len):
            sig.append(((i * 7 + j * 13 + 42) % 1000) / 100.0)
        signals.append(sig)

    total_distance = 0.0
    max_distance = 0.0

    for i in range(n):
        sig_a = signals[2 * i]
        sig_b = signals[2 * i + 1]

        # Compute sum of squared differences
        sq_sum = 0.0
        for j in range(sig_len):
            diff = sig_a[j] - sig_b[j]
            sq_sum += diff * diff

        # Newton's method sqrt (15 iterations)
        if sq_sum > 0.0:
            x = sq_sum
            for _ in range(15):
                x = (x + sq_sum / x) * 0.5
            dist = x
        else:
            dist = 0.0

        total_distance += dist
        if dist > max_distance:
            max_distance = dist

    return (total_distance, max_distance)
