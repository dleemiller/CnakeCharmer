"""Bitstream float decoder variants."""

from __future__ import annotations

import struct


class Flags:
    Coord = 1 << 0
    NoScale = 1 << 1
    CellCoord = 1 << 2
    Normal = 1 << 3
    CellCoordIntegral = 1 << 4


class FloatDecoder:
    def __init__(self, prop):
        self.prop = prop
        self._bits = prop.bits
        self._low = prop.low
        self._high = prop.high

        if prop.flags & Flags.Coord:
            self._fn = self._decode_coord
        elif prop.flags & Flags.NoScale:
            self._fn = self._decode_no_scale
        elif prop.flags & Flags.CellCoord:
            self._fn = self._decode_cell_coord
        elif prop.flags & Flags.Normal:
            self._fn = self._decode_normal
        elif prop.flags & Flags.CellCoordIntegral:
            self._fn = self._decode_cell_coord_integral
        else:
            self._fn = self._decode_default

    def decode(self, stream):
        return self._fn(stream)

    def _decode_coord(self, stream):
        has_i = stream.read_numeric_bits(1)
        has_f = stream.read_numeric_bits(1)
        if not (has_i or has_f):
            return 0.0
        s = stream.read_numeric_bits(1)
        i = stream.read_numeric_bits(14) + 1 if has_i else 0
        f = stream.read_numeric_bits(5) if has_f else 0
        v = float(i) + 0.03125 * f
        return -v if s else v

    def _decode_no_scale(self, stream):
        return struct.unpack("f", stream.read_bits(32))[0]

    def _decode_normal(self, stream):
        s = stream.read_numeric_bits(1)
        low_bits = stream.read_numeric_bits(11)
        b = bytearray([0, 0, (low_bits & 0x0000FF00) >> 8, low_bits & 0x000000FF])
        v = struct.unpack("f", b)[0]
        v *= 4.885197850512946e-4
        return -v if s else v

    def _decode_cell_coord(self, stream):
        v = stream.read_numeric_bits(self._bits)
        return float(v) + 0.01325 * stream.read_numeric_bits(5)

    def _decode_cell_coord_integral(self, stream):
        v = stream.read_numeric_bits(self._bits)
        return float(v)

    def _decode_default(self, stream):
        t = stream.read_numeric_bits(self._bits)
        f = float(t) / (1 << (self._bits - 1))
        return f * (self._high - self._low) + self._low


def mk(prop):
    return FloatDecoder(prop)
