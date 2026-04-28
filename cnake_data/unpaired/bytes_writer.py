"""Primitive serializer into a byte buffer."""

from __future__ import annotations

import struct


class BytesWriter:
    def __init__(self):
        self.buf = bytearray()

    def write_char(self, value: int):
        self.buf.append(value & 0xFF)

    def write_int(self, value: int):
        negative = value < 0
        mag = -value if negative else value
        payload = mag.to_bytes((mag.bit_length() + 7) // 8 or 1, "little", signed=False)
        size = len(payload) | (0b10000 if negative else 0)
        self.write_char(size)
        self.buf.extend(payload)

    def write_uint(self, value: int):
        payload = value.to_bytes((value.bit_length() + 7) // 8 or 1, "little", signed=False)
        self.write_char(len(payload))
        self.buf.extend(payload)

    def write_size_t(self, value: int):
        self.buf.extend(struct.pack("<Q", value))

    def write_float(self, value: float):
        self.buf.extend(struct.pack("<d", value))

    def write_bytes(self, value: bytes):
        self.write_size_t(len(value))
        self.buf.extend(value)

    def write_str(self, value: str):
        self.write_bytes(value.encode())

    def build(self) -> bytes:
        return bytes(self.buf)
