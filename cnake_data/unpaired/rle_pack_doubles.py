import math
import struct


def array_is_empty(data):
    """True when all entries are NaN."""
    for x in data:
        if not math.isnan(x):
            return False
    return True


def pack(values):
    """Run-length pack doubles using (count, mode, payload) tuples.

    mode=1: repeated single value
    mode=0: literal block
    """
    if not values:
        return []

    out = []
    literals = []

    def flush_literals():
        if literals:
            out.append((len(literals), 0, tuple(literals)))
            literals.clear()

    i = 0
    n = len(values)
    while i < n:
        v = values[i]
        j = i + 1
        while j < n and values[j] == v:
            j += 1
        run = j - i

        if run > 1:
            flush_literals()
            out.append((run, 1, v))
        else:
            literals.append(v)
        i = j

    flush_literals()
    return out


def unpack(encoded):
    out = []
    for count, mode, payload in encoded:
        if mode == 1:
            out.extend([payload] * count)
        else:
            out.extend(payload)
    return out


def pack_to_bytes(values):
    """Optional byte serialization: simple tagged format for packed tuples."""
    chunks = []
    for count, mode, payload in pack(values):
        chunks.append(struct.pack("<IB", count, mode))
        if mode == 1:
            chunks.append(struct.pack("<d", payload))
        else:
            chunks.append(struct.pack(f"<{count}d", *payload))
    return b"".join(chunks)
