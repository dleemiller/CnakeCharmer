"""Chain-multiply n 2x2 matrices and return the trace.

Keywords: matrix, 2x2, multiply, chain, trace, operator overloading, numerical, benchmark
"""

from cnake_data.benchmarks import python_benchmark


class Mat2x2:
    """2x2 matrix with __mul__ for matrix multiplication."""

    __slots__ = ("a", "b", "c", "d")

    def __init__(self, a, b, c, d):
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def __mul__(self, other):
        return Mat2x2(
            self.a * other.a + self.b * other.c,
            self.a * other.b + self.b * other.d,
            self.c * other.a + self.d * other.c,
            self.c * other.b + self.d * other.d,
        )

    def trace(self):
        return self.a + self.d


@python_benchmark(args=(100000,))
def matrix2x2_power(n: int) -> float:
    """Generate n 2x2 matrices and chain-multiply them, returning the trace.

    Each matrix has small entries to avoid overflow. The chain product
    M_0 * M_1 * ... * M_{n-1} is computed left-to-right using __mul__.

    Args:
        n: Number of matrices to chain-multiply.

    Returns:
        Trace of the resulting product matrix.
    """
    # Start with identity
    result = Mat2x2(1.0, 0.0, 0.0, 1.0)

    for i in range(n):
        h1 = ((i * 2654435761 + 1) >> 8) & 0xFFFF
        h2 = ((i * 1103515245 + 3) >> 8) & 0xFFFF
        h3 = ((i * 2246822519 + 5) >> 8) & 0xFFFF
        h4 = ((i * 6364136223846793005 + 7) >> 16) & 0xFFFF

        # Small entries near identity to keep values bounded
        # Diagonal near 1.0, off-diagonal near 0
        ma = 1.0 + (h1 % 21 - 10) / 1000.0
        mb = (h2 % 21 - 10) / 1000.0
        mc = (h3 % 21 - 10) / 1000.0
        md = 1.0 + (h4 % 21 - 10) / 1000.0

        m = Mat2x2(ma, mb, mc, md)
        result = result * m

    return result.trace()
