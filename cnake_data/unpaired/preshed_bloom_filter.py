import hashlib
import math
import struct
from array import array


def calculate_size_and_hash_count(members, error_rate):
    base = math.log(1 / (2 ** math.log(2)))
    bit_count = math.ceil((members * math.log(error_rate)) / base)
    hash_count = math.floor((bit_count / members) * math.log(2))
    return bit_count, hash_count


def _hash_pair(item, seed=0):
    data = struct.pack("<qI", int(item), int(seed) & 0xFFFFFFFF)
    digest = hashlib.blake2b(data, digest_size=16).digest()
    a = int.from_bytes(digest[:8], "little", signed=False)
    b = int.from_bytes(digest[8:], "little", signed=False)
    return a, b


class BloomFilter:
    def __init__(self, size=2**10, hash_funcs=23, seed=0):
        self.length = int(size)
        self.hcount = int(hash_funcs)
        self.seed = int(seed)
        self.bitfield = bytearray((self.length + 7) // 8)

    @classmethod
    def from_error_rate(cls, members, error_rate=1e-4):
        return cls(*calculate_size_and_hash_count(members, error_rate))

    def _indices(self, item):
        a, b = _hash_pair(item, self.seed)
        for hiter in range(self.hcount):
            yield (a + hiter * b) % self.length

    def add(self, item):
        for hv in self._indices(item):
            self.bitfield[hv // 8] |= 1 << (hv % 8)

    def __contains__(self, item):
        for hv in self._indices(item):
            if not (self.bitfield[hv // 8] & (1 << (hv % 8))):
                return False
        return True

    def to_bytes(self):
        header = array("L", [self.hcount, self.length, self.seed])
        return header.tobytes() + bytes(self.bitfield)

    def from_bytes(self, byte_string):
        py = array("L")
        hdr_sz = 3 * py.itemsize
        py.frombytes(byte_string[:hdr_sz])
        self.hcount, self.length, self.seed = int(py[0]), int(py[1]), int(py[2])
        self.bitfield = bytearray(byte_string[hdr_sz:])
        return self
