"""Dynamic numeric array growth pattern with amortized reallocation."""

from __future__ import annotations


class DynamicDoubleArray:
    def __init__(self, initial_capacity: int = 16):
        cap = max(1, int(initial_capacity))
        self.capacity = cap
        self.length = 0
        self.data = [0.0] * cap

    def append(self, value: float) -> None:
        if self.length >= self.capacity:
            self.capacity *= 2
            self.data.extend([0.0] * (self.capacity - len(self.data)))
        self.data[self.length] = float(value)
        self.length += 1

    def to_list(self) -> list[float]:
        return self.data[: self.length]


def memory(values: list[float]) -> list[float]:
    arr = DynamicDoubleArray(max(16, len(values)))
    for v in values:
        arr.append(v)
    return arr.to_list()
