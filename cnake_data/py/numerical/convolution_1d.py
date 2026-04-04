"""
1D convolution of a deterministic signal with a fixed kernel.

Keywords: numerical, convolution, signal processing, 1D, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(100000,))
def convolution_1d(n: int) -> list:
    """Compute 1D convolution of a signal with a 5-element kernel.

    Signal is generated as signal[i] = (i*7+3) % 100 / 10.0.
    Kernel is [0.1, 0.2, 0.4, 0.2, 0.1].
    Returns valid convolution (length n - 4).

    Args:
        n: Length of the input signal.

    Returns:
        List of floats representing the convolved output.
    """
    signal = [(i * 7 + 3) % 100 / 10.0 for i in range(n)]
    kernel = [0.1, 0.2, 0.4, 0.2, 0.1]
    k_len = len(kernel)
    out_len = n - k_len + 1

    result = []
    for i in range(out_len):
        s = 0.0
        for j in range(k_len):
            s += signal[i + j] * kernel[j]
        result.append(s)

    return result
