"""LFSR generators and Geffe combiner."""

from __future__ import annotations


class LFSR:
    def __init__(self, size, taps, start_state=None, fill=1):
        self.taps = taps
        if isinstance(start_state, list) and len(start_state) == size:
            self.state = list(start_state)
        else:
            self.state = [fill for _ in range(size)]

    def __next__(self):
        new_bit = 0
        for t in self.taps:
            new_bit ^= self.state[t]
        for i in range(1, len(self.state)):
            self.state[i - 1] = self.state[i]
        self.state[-1] = new_bit
        return new_bit

    def get_octal_state(self):
        bits = "".join(str(next(self)) for _ in range(8))
        return int(bits, 2)


class Geffe:
    def __init__(self, x, y, s):
        self.x = x
        self.y = y
        self.s = s

    def __next__(self):
        x_new = next(self.x)
        y_new = next(self.y)
        s_new = next(self.s)
        return (s_new & x_new) ^ ((1 ^ s_new) & y_new)
