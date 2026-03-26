"""Instance normalization per-channel.

channels=16, spatial=n/16. Normalize each channel independently.

Keywords: instance_norm, normalization, neural network, tensor, f32, benchmark
"""

import math

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(1000000,))
def instance_norm(n: int) -> float:
    """Instance normalization per-channel, return sum of output.

    Args:
        n: Total input size (channels * spatial).

    Returns:
        Sum of normalized output.
    """
    channels = 16
    spatial = n // channels
    eps = 1e-5

    # Generate input
    data = [math.sin(i * 0.01) * 10.0 for i in range(n)]

    total = 0.0
    for c in range(channels):
        offset = c * spatial

        # Compute mean
        mean = 0.0
        for s in range(spatial):
            mean += data[offset + s]
        mean /= spatial

        # Compute variance
        var = 0.0
        for s in range(spatial):
            diff = data[offset + s] - mean
            var += diff * diff
        var /= spatial

        # Normalize
        inv_std = 1.0 / math.sqrt(var + eps)
        for s in range(spatial):
            total += (data[offset + s] - mean) * inv_std

    return total
