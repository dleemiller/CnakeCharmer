"""1D convolution of a signal with a smoothing kernel.

Keywords: convolution, 1d, signal processing, neural network, cnn
"""

import math

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(1000000,))
def conv1d(n: int) -> float:
    """Convolve signal[i]=sin(i*0.01)*100 with kernel [1,2,3,4,3,2,1]/16.

    Args:
        n: Signal length.

    Returns:
        Sum of convolution output.
    """
    kernel = [1.0 / 16.0, 2.0 / 16.0, 3.0 / 16.0, 4.0 / 16.0, 3.0 / 16.0, 2.0 / 16.0, 1.0 / 16.0]
    k_len = len(kernel)
    out_len = n - k_len + 1
    total = 0.0
    for i in range(out_len):
        s = 0.0
        for j in range(k_len):
            s += math.sin((i + j) * 0.01) * 100.0 * kernel[j]
        total += s
    return total
