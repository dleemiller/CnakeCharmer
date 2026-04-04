"""Apply linear transforms to values using callable objects.

Keywords: numerical, callable, transform, linear, benchmark
"""

from cnake_data.benchmarks import python_benchmark


class LinearTransform:
    """Callable linear transform: result = scale * x + offset."""

    def __init__(self, scale, offset):
        self.scale = scale
        self.offset = offset

    def __call__(self, x):
        return self.scale * x + self.offset


@python_benchmark(args=(200000,))
def callable_transform(n: int) -> float:
    """Create transform objects and apply to n values, summing results.

    Args:
        n: Number of values to transform.

    Returns:
        Sum of all transformed values.
    """
    # Create a set of transforms deterministically
    transforms = []
    for i in range(16):
        scale = ((i * 2654435761 + 17) % 1000) / 100.0 - 5.0
        offset = ((i * 1103515245 + 12345) % 1000) / 50.0 - 10.0
        transforms.append(LinearTransform(scale, offset))

    total = 0.0
    for i in range(n):
        val = ((i * 1664525 + 1013904223) % 1000000) / 1000.0
        t = transforms[i & 15]
        total += t(val)

    return total
