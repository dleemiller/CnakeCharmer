from __future__ import annotations


class GrowableDoubles:
    def __init__(self, size: int):
        self.data = [0.0] * size
        self.length = size
        self.it = 0

    def resize(self, new_size: int) -> None:
        if new_size > self.length:
            self.data.extend([0.0] * (new_size - self.length))
            self.length = new_size

    def insert(self, value: float, re_size: bool = False, new_size: int = -1) -> None:
        if self.it < self.length:
            self.data[self.it] = value
            self.it += 1
            return
        if re_size and new_size > 0:
            self.resize(new_size)
            self.data[self.it] = value
            self.it += 1
            return
        raise TypeError("buffer full")

    def values(self) -> list[float]:
        return self.data[: self.it]
