"""Reinterpret integer bits as float via type punning and sum finite results.

Keywords: union, type punning, int, float, numerical, benchmark
"""

import math
import struct

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(100000,))
def union_int_float(n: int) -> float:
    """Reinterpret n hash-derived ints as floats, sum finite values.

    Uses struct pack/unpack to emulate C union type punning.

    Args:
        n: Number of integers to reinterpret.

    Returns:
        Sum of all finite float reinterpretations.
    """
    total = 0.0
    for i in range(n):
        # Deterministic hash-based integer
        h = (i * 2654435761) & 0xFFFFFFFF
        # Reinterpret 4-byte int as float (type punning)
        raw = struct.pack("I", h)
        f = struct.unpack("f", raw)[0]
        if math.isfinite(f):
            total += f
    return total
