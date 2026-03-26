"""1D max pooling.

Keywords: max pool, pooling, neural network, downsampling
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(3000000,))
def max_pool_1d(n: int) -> int:
    """Max pool with kernel=4, stride=4 and return sum of pooled values.

    signal[i] = (i * 31 + 17) % 1000

    Args:
        n: Signal length.

    Returns:
        Sum of pooled values.
    """
    total = 0
    num_pools = n // 4
    for i in range(num_pools):
        base = i * 4
        max_val = (base * 31 + 17) % 1000
        for j in range(1, 4):
            v = ((base + j) * 31 + 17) % 1000
            if v > max_val:
                max_val = v
        total += max_val
    return total
