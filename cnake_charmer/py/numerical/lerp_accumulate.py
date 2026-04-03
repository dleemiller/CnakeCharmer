"""Linear interpolation accumulation.

Keywords: numerical, interpolation, lerp, accumulation, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(1.25, 5.5, 250000, 0.00001))
def lerp_accumulate(a0: float, b0: float, steps: int, delta: float) -> float:
    """Accumulate interpolated values on evolving endpoints."""
    total = 0.0
    a = a0
    b = b0
    for i in range(steps):
        t = (i % 1024) * 0.0009765625
        total += a + t * (b - a)
        a += delta
        b -= delta * 0.5
    return total
