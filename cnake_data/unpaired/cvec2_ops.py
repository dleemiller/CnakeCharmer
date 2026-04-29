"""2D vector utility class."""

from __future__ import annotations

from math import sqrt


class cvec2:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def __iter__(self):
        return iter((self.x, self.y))

    def __getitem__(self, key: int):
        if key == 0:
            return self.x
        if key == 1:
            return self.y
        raise IndexError

    def __setitem__(self, key: int, value: float):
        if key == 0:
            self.x = value
        elif key == 1:
            self.y = value
        else:
            raise IndexError

    def __sub__(self, vec):
        return cvec2(self.x - vec.x, self.y - vec.y)

    def __add__(self, vec):
        return cvec2(self.x + vec.x, self.y + vec.y)

    def __mul__(self, num):
        if isinstance(num, (float, int)):
            return cvec2(self.x * num, self.y * num)
        if isinstance(num, cvec2):
            return cvec2(self.x * num.x, self.y * num.y)
        return NotImplemented

    def __truediv__(self, num: float):
        return cvec2(self.x / num, self.y / num)

    def __pow__(self, _num, _mod=None):
        return cvec2(self.x * self.x, self.y * self.y)

    def mag(self):
        return sqrt(self.x * self.x + self.y * self.y)

    def normalize(self):
        mag = self.mag()
        return cvec2(self.x / mag, self.y / mag)
