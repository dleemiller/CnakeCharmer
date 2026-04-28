import os
import struct

FOO_STRUCT = struct.Struct("ii")


class Foo:
    def __init__(self, bar=0, baz=0):
        self.bar = int(bar)
        self.baz = int(baz)

    @property
    def as_bytes(self):
        return FOO_STRUCT.pack(self.bar, self.baz)

    @classmethod
    def from_bytes(cls, foo_bytes):
        bar, baz = FOO_STRUCT.unpack(foo_bytes[: FOO_STRUCT.size])
        return cls(bar=bar, baz=baz)

    def __len__(self):
        return FOO_STRUCT.size

    def __repr__(self):
        return f"Foo({self.bar}, {self.baz})"


def foo_from_mmap(file_name):
    with open(file_name, "r+b") as f:
        data = f.read(FOO_STRUCT.size)
    return Foo.from_bytes(data)


def foo_to_mmap(file_name, foo):
    os.makedirs(os.path.dirname(file_name) or ".", exist_ok=True)
    with open(file_name, "wb") as f:
        f.write(foo.as_bytes)
