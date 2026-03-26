"""Global average pooling.

channels=64, spatial=n/64. Return sum of channel means.

Keywords: global_avg_pool, pooling, neural network, tensor, f32, benchmark
"""

import math

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(1000000,))
def global_avg_pool(n: int) -> float:
    """Global average pool, return sum of channel means.

    Args:
        n: Total input size (channels * spatial).

    Returns:
        Sum of channel means.
    """
    channels = 64
    spatial = n // channels

    # Generate input
    data = [math.sin(i * 0.01) * 10.0 for i in range(n)]

    # Global average pool: mean per channel
    total = 0.0
    for c in range(channels):
        offset = c * spatial
        channel_sum = 0.0
        for s in range(spatial):
            channel_sum += data[offset + s]
        total += channel_sum / spatial

    return total
