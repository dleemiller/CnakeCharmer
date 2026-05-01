"""Dictionary-backed weighted sum benchmark helpers."""

from __future__ import annotations

import time


class WeightedDictModel:
    def __init__(self, a, b, c, d, e, f):
        self.dct = {"a": a, "b": b, "c": c, "d": d, "e": e, "f": f}

    def calculate(self, test_dict: dict[str, float]) -> float:
        return (
            self.dct["a"] * test_dict["a"]
            + self.dct["b"] * test_dict["b"]
            + self.dct["c"] * test_dict["c"]
            + self.dct["d"] * test_dict["d"]
            + self.dct["e"] * test_dict["e"]
            + self.dct["f"] * test_dict["f"]
        )


def benchmark_single(test_dict: dict[str, float], iters: int = 1_000_000) -> float:
    model = WeightedDictModel(1, 2, 3, 4, 5, 6)
    before = time.perf_counter()
    total = 0.0
    for _ in range(iters):
        total += model.calculate(test_dict)
    after = time.perf_counter()
    print(f"weighted_dot_single elapsed {after - before:.7f}")
    return total


def benchmark_many(test_dict: dict[str, float], iters: int = 1_000_000) -> float:
    models = [WeightedDictModel(1, 2, 3, 4, 5, 6) for _ in range(iters)]
    before = time.perf_counter()
    total = 0.0
    for i in range(iters):
        total += models[i].calculate(test_dict)
    after = time.perf_counter()
    print(f"weighted_dot_many elapsed {after - before:.7f}")
    return total
