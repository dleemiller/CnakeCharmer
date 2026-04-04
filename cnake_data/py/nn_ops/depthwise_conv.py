"""Depthwise 1D convolution.

Each channel convolved independently. channels=16, kernel=3.

Keywords: depthwise_conv, convolution, neural network, tensor, f32, benchmark
"""

import math

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(1000000,))
def depthwise_conv(n: int) -> float:
    """Depthwise 1D convolution, return sum of output.

    Args:
        n: Total input size (channels * spatial).

    Returns:
        Sum of convolution output.
    """
    channels = 16
    spatial = n // channels

    # Generate input [channels x spatial]
    inp = [math.sin(i * 0.01) * 10.0 for i in range(n)]

    # Generate kernel [channels x 3]
    kernel_size = 3
    kernel = [0.0] * (channels * kernel_size)
    for c in range(channels):
        for k in range(kernel_size):
            kernel[c * kernel_size + k] = math.sin((c * kernel_size + k) * 0.5) * 0.5

    # Depthwise conv: each channel independently
    out_spatial = spatial - kernel_size + 1
    total = 0.0
    for c in range(channels):
        inp_offset = c * spatial
        for s in range(out_spatial):
            val = 0.0
            for k in range(kernel_size):
                val += inp[inp_offset + s + k] * kernel[c * kernel_size + k]
            total += val

    return total
