"""Contiguous byte-buffer transport helpers with read/write cursor logic."""

from __future__ import annotations


class CyBuffer:
    def __init__(self, buf_size: int):
        self.buf = bytearray(buf_size)
        self.buf_size = buf_size
        self.cur = 0
        self.data_size = 0

    def _ensure(self, extra: int) -> None:
        need = self.cur + extra
        if need <= self.buf_size:
            return
        new_size = max(need, self.buf_size * 2)
        self.buf.extend(b"\x00" * (new_size - self.buf_size))
        self.buf_size = new_size

    def write(self, data: bytes) -> None:
        self._ensure(len(data))
        self.buf[self.cur : self.cur + len(data)] = data
        self.cur += len(data)
        self.data_size = max(self.data_size, self.cur)

    def seek(self, pos: int) -> None:
        if pos < 0 or pos > self.data_size:
            raise ValueError("invalid seek")
        self.cur = pos

    def read(self, n: int) -> bytes:
        end = min(self.cur + n, self.data_size)
        out = bytes(self.buf[self.cur : end])
        self.cur = end
        return out
