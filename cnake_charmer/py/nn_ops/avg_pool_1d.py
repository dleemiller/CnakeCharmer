"""Average pooling 1D on a float tensor.

Average pooling with kernel=4, stride=4.

Keywords: avg_pool, pooling, neural network, tensor, f32, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(5000000,))
def avg_pool_1d(n: int) -> float:
    """Allocate f32 signal, apply avg pool with kernel=4 stride=4, return sum of pooled.

    Args:
        n: Signal length (number of float elements).

    Returns:
        Sum of pooled values.
    """
    # Generate signal
    signal = [(i * 31 + 17) % 1000 / 10.0 for i in range(n)]

    # Average pool kernel=4, stride=4
    out_len = n // 4
    total = 0.0
    for i in range(out_len):
        base = i * 4
        avg = (signal[base] + signal[base + 1] + signal[base + 2] + signal[base + 3]) * 0.25
        total += avg

    return total
