"""Clamp hash-derived values to a range and sum the results.

Keywords: clamp, numerical, cpdef, standalone function, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


def clamp(val: float, lo: float, hi: float) -> float:
    """Clamp val to the range [lo, hi]."""
    if val < lo:
        return lo
    if val > hi:
        return hi
    return val


@python_benchmark(args=(100000,))
def cpdef_clamp_sum(n: int) -> float:
    """Clamp n hash-derived values to [-5.0, 5.0] and sum results.

    Args:
        n: Number of values to clamp and sum.

    Returns:
        Sum of all clamped values.
    """
    total = 0.0
    for i in range(n):
        h = (i * 2654435761) & 0xFFFFFFFF
        val = (h / 4294967295.0) * 20.0 - 10.0
        total += clamp(val, -5.0, 5.0)
    return total
