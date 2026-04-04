"""Class wrapping paired numeric buffers with aggregation methods.

Keywords: algorithms, class, object methods, buffers, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


class DoubleList:
    def __init__(self, n: int, scale: float):
        self.left = [((i * 17 + 3) % 100) * scale for i in range(n)]
        self.right = [((i * 19 + 5) % 100) * scale for i in range(n)]

    def blend(self, alpha: float, beta: float) -> float:
        total = 0.0
        for a, b in zip(self.left, self.right, strict=False):
            total += a * alpha + b * beta
        return total


@python_benchmark(args=(700, 0.17, 900, 0.61, 0.39))
def double_list_accumulator(n: int, scale: float, rounds: int, alpha: float, beta: float) -> tuple:
    obj = DoubleList(n, scale)
    total = 0.0
    peak = 0.0
    for r in range(rounds):
        v = obj.blend(alpha + (r & 3) * 0.01, beta - (r & 1) * 0.02)
        total += v
        if v > peak:
            peak = v
    return (total, peak, obj.left[n // 2])
