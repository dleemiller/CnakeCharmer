"""1D Gaussian blur convolution.

Keywords: grpo, signal processing, convolution, gaussian, filter, benchmark
"""

import math

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(100000,))
def gauss_blur_1d(n: int) -> tuple:
    """Apply 1D Gaussian blur to a deterministic signal.

    Generates a signal and convolves it with a Gaussian kernel of radius 5.

    Returns (sum of output, max of output, min of output).

    Args:
        n: Length of the input signal.

    Returns:
        Tuple of (sum_output, max_output, min_output).
    """
    # Generate signal
    signal = [0.0] * n
    for i in range(n):
        signal[i] = math.sin(i * 0.01) + 0.5 * math.sin(i * 0.03)

    # Gaussian kernel (radius=5, sigma=1.5)
    radius = 5
    sigma = 1.5
    kernel_size = 2 * radius + 1
    kernel = [0.0] * kernel_size
    kernel_sum = 0.0
    for i in range(kernel_size):
        x = i - radius
        kernel[i] = math.exp(-0.5 * (x / sigma) ** 2)
        kernel_sum += kernel[i]

    # Normalize
    for i in range(kernel_size):
        kernel[i] /= kernel_sum

    # Convolve (boundary: clamp)
    output = [0.0] * n
    for i in range(n):
        val = 0.0
        for k in range(kernel_size):
            j = i + k - radius
            if j < 0:
                j = 0
            elif j >= n:
                j = n - 1
            val += signal[j] * kernel[k]
        output[i] = val

    sum_out = 0.0
    max_out = output[0]
    min_out = output[0]
    for v in output:
        sum_out += v
        if v > max_out:
            max_out = v
        if v < min_out:
            min_out = v

    return (round(sum_out, 4), round(max_out, 6), round(min_out, 6))
