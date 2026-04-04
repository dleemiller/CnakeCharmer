"""Chain affine transforms with not-None enforcement.

Keywords: numerical, transform, not none, extension type, affine, benchmark
"""

from cnake_data.benchmarks import python_benchmark


class Transform:
    """Affine transform: y = scale * x + offset."""

    def __init__(self, scale, offset):
        self.scale = scale
        self.offset = offset

    def apply(self, other):
        """Compose self(other(x)) = self.scale*(other.scale*x + other.offset) + self.offset."""
        if other is None:
            raise TypeError("other must not be None")
        return Transform(
            self.scale * other.scale,
            self.scale * other.offset + self.offset,
        )


@python_benchmark(args=(50000,))
def not_none_transform(n: int) -> float:
    """Chain n transforms, return final scale + offset.

    Args:
        n: Number of transforms to chain.

    Returns:
        Sum of final scale and offset.
    """
    result = Transform(1.0, 0.0)
    for i in range(n):
        s = ((i * 2654435761) % 1000) / 500.0
        o = ((i * 1664525 + 1013904223) % 1000) / 500.0
        t = Transform(s, o)
        # Compose and re-normalize to prevent overflow
        result = result.apply(t)
        norm = abs(result.scale) + abs(result.offset) + 1.0
        result = Transform(result.scale / norm, result.offset / norm)
    return result.scale + result.offset
