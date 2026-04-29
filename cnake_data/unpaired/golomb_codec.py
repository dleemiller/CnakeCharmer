"""Golomb encoder/decoder utilities."""

from __future__ import annotations

import math


class GolombCodec:
    def __init__(self, m):
        self.m = int(m)
        self.c = int(math.ceil(math.log2(self.m)))
        self.div = int(2**self.c - self.m)
        self.buffer = []
        self.byte = 0
        self.pointer = 7

    def to_golomb(self, val):
        q = val // self.m
        r = val % self.m
        for _ in range(q):
            self._write_bit(1)
        self._write_bit(0)

        if r < self.div:
            self._write_n_bits(r, self.c - 1)
        else:
            self._write_n_bits(r + self.div, self.c)

    def get_bytes(self):
        if self.pointer != 7:
            self._write_n_bits(0xFF, self.pointer + 1)
        b = bytes(self.buffer)
        self.buffer = []
        return b

    def _write_bit(self, b):
        self.byte |= (b & 0x01) << self.pointer
        if self.pointer > 0:
            self.pointer -= 1
            return
        self.buffer.append(self.byte)
        self.pointer = 7
        self.byte = 0

    def _write_n_bits(self, b, length):
        i = length - 1
        while i >= 0:
            bit = (b >> (i % 32)) & 0x01
            self._write_bit(bit)
            i -= 1


class ReadGolomb:
    def __init__(self, m, raw_data):
        self.buffer = list(raw_data)
        self.pointer = -1
        self.next_read = 0
        self.end = 0
        self.m = m
        self.c = int(math.ceil(math.log2(self.m)))
        self.div = int(2**self.c - self.m)

    def read_values(self):
        values = []
        while self.end == 0 or self.pointer > 0:
            val = self._read_val()
            if self.pointer >= -1:
                values.append(val)
        return values

    def _read_bit(self):
        if self.pointer < 0:
            if self.next_read >= len(self.buffer):
                self.end = 1
                return 0
            self.byte = self.buffer[self.next_read]
            self.next_read += 1
            self.pointer = 7
        bit = (self.byte >> self.pointer) & 0x01
        self.pointer -= 1
        return bit

    def _read_n_bits(self, length):
        value = 0
        for _ in range(length):
            value = (value << 1) | self._read_bit()
        return value

    def _read_val(self):
        q = 0
        while self._read_bit() == 1:
            q += 1
        r = self._read_n_bits(self.c - 1)
        if r >= self.div:
            r = ((r << 1) | self._read_bit()) - self.div
        return q * self.m + r
