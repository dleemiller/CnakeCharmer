"""Use a config object to control iterative computation parameters.

Keywords: algorithms, config, readonly, iteration, tolerance, benchmark
"""

from cnake_data.benchmarks import python_benchmark


class Config:
    """Configuration with read-only max_iter and tolerance."""

    __slots__ = ("_max_iter", "_tolerance")

    def __init__(self, max_iter, tolerance):
        object.__setattr__(self, "_max_iter", max_iter)
        object.__setattr__(self, "_tolerance", tolerance)

    @property
    def max_iter(self):
        return self._max_iter

    @property
    def tolerance(self):
        return self._tolerance

    def __setattr__(self, name, value):
        raise AttributeError("Config attributes are read-only")


@python_benchmark(args=(100000,))
def config_lookup(n: int) -> float:
    """Run n independent convergence computations controlled by Config objects.

    Each computation iterates until convergence or max_iter, accumulating results.

    Args:
        n: Number of computations to run.

    Returns:
        Sum of final values from all computations.
    """
    # Create a set of configs
    configs = []
    for i in range(8):
        max_iter = ((i * 2654435761 + 17) % 50) + 10
        tolerance = 1.0 / (((i * 1103515245 + 12345) % 100) + 10)
        configs.append(Config(max_iter, tolerance))

    total = 0.0
    for i in range(n):
        cfg = configs[i & 7]
        # Simple iterative computation: repeatedly halve and add noise
        x = ((i * 1664525 + 1013904223) % 100000) / 100.0
        for j in range(cfg.max_iter):
            prev = x
            x = x * 0.5 + ((j * 214013 + i) % 100) / 200.0
            if abs(x - prev) < cfg.tolerance:
                break
        total += x

    return total
