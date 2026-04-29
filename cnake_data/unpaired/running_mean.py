"""Online weighted running-mean accumulator."""

from __future__ import annotations


class RunningMean:
    def __init__(self, mean=0.0, n=0.0):
        self.mean = float(mean)
        self.n = float(n)

    def update(self, x, w=1.0):
        self.n += w
        if self.n > 0:
            self.mean += w * (x - self.mean) / self.n
        return self

    def revert(self, x, w=1.0):
        self.n -= w
        if self.n < 0:
            raise ValueError("Cannot go below 0")
        if self.n == 0:
            self.mean = 0.0
        else:
            self.mean -= w * (x - self.mean) / self.n
        return self

    def get(self):
        return self.mean

    def __add__(self, other):
        result = RunningMean()
        result.n = self.n + other.n
        if result.n == 0:
            result.mean = 0.0
        else:
            result.mean = (self.n * self.mean + other.n * other.mean) / result.n
        return result
