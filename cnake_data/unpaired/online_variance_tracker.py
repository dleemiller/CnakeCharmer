"""Online variance tracker using Welford-style updates for vector streams."""

from __future__ import annotations


class OnlineVariance:
    def __init__(self, length: int):
        self.length = int(length)
        self.count = 0
        self.mean = [0.0] * self.length
        self.m2 = [0.0] * self.length

    def clear(self) -> None:
        self.count = 0
        for i in range(self.length):
            self.mean[i] = 0.0
            self.m2[i] = 0.0

    def update(self, values: list[float]) -> None:
        if len(values) != self.length:
            raise ValueError("length mismatch")

        self.count += 1
        n = self.count

        for i in range(self.length):
            x = float(values[i])
            delta = x - self.mean[i]
            self.mean[i] += delta / n
            delta2 = x - self.mean[i]
            self.m2[i] += delta * delta2

    def variance(self) -> list[float]:
        if self.count < 2:
            return [0.0] * self.length
        denom = self.count - 1
        return [v / denom for v in self.m2]
