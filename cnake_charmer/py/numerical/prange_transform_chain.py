"""Parallel chain of transforms (scale, shift, clamp) on n values.

Keywords: numerical, transform, chain, parallel, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(1000000,))
def prange_transform_chain(n: int) -> float:
    """Apply scale, shift, clamp transforms to n values, return sum.

    Each value starts as a hash-derived float in [0, 1), then:
    1. Scale by 2.5
    2. Shift by -0.75
    3. Clamp to [0.0, 1.5]

    Args:
        n: Number of values to transform.

    Returns:
        Sum of all transformed values.
    """
    total = 0.0
    for i in range(n):
        val = (i * 2654435761 & 0xFFFFFFFF) / 4294967296.0
        # scale
        val = val * 2.5
        # shift
        val = val - 0.75
        # clamp
        if val < 0.0:
            val = 0.0
        elif val > 1.5:
            val = 1.5
        total += val

    return total
