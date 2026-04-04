"""Sum traces of dot products from Matrix pairs with not-None checks.

Keywords: numerical, matrix, not none, extension type, trace, benchmark
"""

from cnake_data.benchmarks import python_benchmark


class Matrix:
    """Simple 2x2 matrix stored as four doubles."""

    def __init__(self, a, b, c, d):
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def dot(self, other):
        """Multiply self by other, return new Matrix."""
        if other is None:
            raise TypeError("other must not be None")
        return Matrix(
            self.a * other.a + self.b * other.c,
            self.a * other.b + self.b * other.d,
            self.c * other.a + self.d * other.c,
            self.c * other.b + self.d * other.d,
        )

    def trace(self):
        return self.a + self.d


@python_benchmark(args=(10000,))
def not_none_matrix_sum(n: int) -> float:
    """Create n matrix pairs, multiply them, sum all traces.

    Args:
        n: Number of matrix pairs to process.

    Returns:
        Sum of traces of all matrix products.
    """
    total = 0.0
    for i in range(n):
        a = ((i * 2654435761) % 1000) / 100.0
        b = ((i * 1664525 + 1013904223) % 1000) / 100.0
        c = ((i * 1103515245 + 12345) % 1000) / 100.0
        d = ((i * 214013 + 2531011) % 1000) / 100.0
        m1 = Matrix(a, b, c, d)

        a2 = ((i * 1566083941 + 1) % 1000) / 100.0
        b2 = ((i * 2053540636 + 7) % 1000) / 100.0
        c2 = ((i * 1654435769 + 13) % 1000) / 100.0
        d2 = ((i * 1013904243 + 19) % 1000) / 100.0
        m2 = Matrix(a2, b2, c2, d2)

        result = m1.dot(m2)
        total += result.trace()
    return total
