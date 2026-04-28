"""Fixed-size block buffer with free-list management."""

from __future__ import annotations


class Buffer:
    def __init__(self, block_size: int):
        self.block_size = block_size
        self.block_count = 0
        self.i_free = 0
        self.data = bytearray()
        self.l_free = []

    def grow(self, block_count: int):
        diff = block_count % 8
        if diff:
            block_count += 8 - diff
        if block_count <= self.block_count:
            return
        self.data.extend(b"\x00" * (self.block_size * (block_count - self.block_count)))
        self.l_free.extend(range(self.block_count, block_count))
        self.block_count = block_count

    def clear(self):
        self.l_free = list(range(self.block_count))
        self.i_free = 0

    def add(self, blocks: bytes, count: int):
        if count > self.block_count - self.i_free:
            self.grow(self.block_count + count)
        indices = []
        for i in range(count):
            start = i * self.block_size
            chunk = blocks[start : start + self.block_size]
            block = self.l_free[self.i_free]
            self.i_free += 1
            dst = block * self.block_size
            self.data[dst : dst + self.block_size] = chunk
            indices.append(block)
        return indices

    def remove(self, indices):
        for idx in indices:
            self.i_free -= 1
            self.l_free[self.i_free] = idx

    def update(self, index: int, blocks: bytes, count: int):
        dst = index * self.block_size
        self.data[dst : dst + self.block_size * count] = blocks[: self.block_size * count]

    def count(self):
        return self.i_free

    def size(self):
        return self.block_size * self.block_count

    def offset_pointer(self, offset: int):
        start = offset * self.block_size
        return memoryview(self.data)[start:]
