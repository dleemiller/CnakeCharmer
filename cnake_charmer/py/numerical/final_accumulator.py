"""Accumulate values with different operations using a final accumulator class.

Keywords: numerical, accumulator, final, operations, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


class Accumulator:
    """Accumulator with typed add, multiply-accumulate, and reset operations."""

    def __init__(self):
        self.value = 0.0
        self.count = 0

    def add(self, x):
        """Add x to the accumulated value."""
        self.value += x
        self.count += 1

    def mul_add(self, x, y):
        """Add x * y to the accumulated value."""
        self.value += x * y
        self.count += 1

    def scale(self, factor):
        """Scale the accumulated value."""
        self.value *= factor

    def get_value(self):
        """Return current accumulated value."""
        return self.value

    def get_count(self):
        """Return operation count."""
        return self.count

    def reset(self):
        """Reset accumulator."""
        self.value = 0.0
        self.count = 0


@python_benchmark(args=(500000,))
def final_accumulator(n: int) -> float:
    """Accumulate n values with mixed operations, return final value.

    Args:
        n: Number of accumulation operations.

    Returns:
        Final accumulated value.
    """
    acc = Accumulator()
    total = 0.0

    for i in range(n):
        op = ((i * 2654435761) >> 4) & 3
        val = ((i * 1664525 + 1013904223) % 10000) / 10000.0

        if op == 0:
            acc.add(val)
        elif op == 1:
            val2 = ((i * 1103515245 + 12345) % 10000) / 10000.0
            acc.mul_add(val, val2)
        elif op == 2:
            acc.scale(0.999)
        else:
            # Periodically harvest and reset
            total += acc.get_value()
            acc.reset()

    total += acc.get_value()
    return total
