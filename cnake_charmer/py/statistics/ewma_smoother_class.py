"""Class-based exponentially weighted moving-average smoothing.

Keywords: statistics, class, ewma, smoothing, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


class EWMASmoother:
    def __init__(self, alpha: float):
        self.alpha = alpha
        self.value = 0.0

    def update(self, x: float) -> float:
        self.value = self.alpha * x + (1.0 - self.alpha) * self.value
        return self.value


@python_benchmark(args=(0.21, 800000, 0.3))
def ewma_smoother_class(alpha: float, steps: int, bias: float) -> tuple:
    sm = EWMASmoother(alpha)
    total = 0.0
    peak = -1e300
    for i in range(steps):
        x = ((i * 37 + 11) % 1000) / 1000.0 + bias
        v = sm.update(x)
        total += v
        if v > peak:
            peak = v
    return (total, peak, sm.value)
