"""Clamped value container with property getter and setter.

Keywords: numerical, property, setter, clamped, bounded, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


class BoundedValue:
    """Value clamped to [lo, hi] range via property setter."""

    def __init__(self, lo, hi):
        self._lo = lo
        self._hi = hi
        self._value = lo

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, v):
        if v < self._lo:
            self._value = self._lo
        elif v > self._hi:
            self._value = self._hi
        else:
            self._value = v


@python_benchmark(args=(100000,))
def property_bounded_value(n: int) -> float:
    """Push n values through a bounded container, return sum.

    Args:
        n: Number of values to push.

    Returns:
        Sum of clamped values.
    """
    bv = BoundedValue(-100.0, 100.0)
    total = 0.0

    for i in range(n):
        seed = (i * 2654435761 + 17) & 0x7FFFFFFF
        raw = (seed % 1000) / 2.0 - 250.0
        bv.value = raw
        total += bv.value

    return total
