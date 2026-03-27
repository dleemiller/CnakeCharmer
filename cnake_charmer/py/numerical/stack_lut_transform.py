"""Apply lookup table transform using a 256-entry table of sin values.

Keywords: numerical, lookup table, sin, transform, benchmark
"""

import math

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(100000,))
def stack_lut_transform(n: int) -> float:
    """Build sin LUT and apply to n values, return sum.

    Args:
        n: Number of values to transform.

    Returns:
        Sum of transformed values.
    """
    lut = [0.0] * 256
    for i in range(256):
        lut[i] = math.sin(i * 0.0245436926)  # ~2*pi/256

    total = 0.0
    for i in range(n):
        idx = ((i * 2654435761) >> 4) & 255
        total += lut[idx]

    return total
