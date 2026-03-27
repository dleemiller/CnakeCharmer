"""Accumulate fixed-point arithmetic values with mixed operations.

Keywords: fixed-point, arithmetic, accumulate, operator overloading, numerical, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


class FixedPoint:
    """Fixed-point number scaled by 1000 (3 decimal places)."""

    __slots__ = ("_val",)
    SCALE = 1000

    def __init__(self, val):
        self._val = int(val)

    def __add__(self, other):
        return FixedPoint(self._val + other._val)

    def __sub__(self, other):
        return FixedPoint(self._val - other._val)

    def __mul__(self, other):
        # Multiply and rescale: (a*S) * (b*S) / S = a*b*S
        return FixedPoint((self._val * other._val) // self.SCALE)

    def __iadd__(self, other):
        self._val += other._val
        return self

    def __neg__(self):
        return FixedPoint(-self._val)

    def __bool__(self):
        return self._val != 0

    @property
    def value(self):
        return self._val


@python_benchmark(args=(100000,))
def fixed_point_accum(n: int) -> int:
    """Accumulate n fixed-point values with mixed arithmetic operations.

    Generates deterministic fixed-point numbers and combines them using
    __add__, __sub__, __mul__, __iadd__, __neg__, and __bool__.

    Args:
        n: Number of values to accumulate.

    Returns:
        Final integer value (scaled by 1000).
    """
    accum = FixedPoint(1000)  # Start at 1.000

    for i in range(n):
        h1 = ((i * 2654435761 + 1) >> 8) & 0xFFFF
        h2 = ((i * 1103515245 + 3) >> 8) & 0xFFFF

        # Generate a fixed-point value in range [-5.000, 5.000]
        raw = h1 % 10001 - 5000
        fp = FixedPoint(raw)

        # Choose operation based on h2
        op = h2 % 5

        if op == 0:
            # __add__
            accum = accum + fp
        elif op == 1:
            # __sub__
            accum = accum - fp
        elif op == 2:
            # __mul__ (scale the value to keep it small)
            # Multiply by a value near 1.0: 0.990 to 1.010
            small = FixedPoint(1000 + (h1 % 21) - 10)
            accum = accum * small
        elif op == 3:
            # __iadd__
            accum += fp
        else:
            # __neg__ and __bool__
            neg_fp = -fp
            accum = accum + neg_fp if neg_fp else accum + FixedPoint(1000)

    return accum.value
