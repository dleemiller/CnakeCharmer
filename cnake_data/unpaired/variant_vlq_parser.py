import struct


def parse_vlq(stream):
    """Parse unsigned variable-length quantity."""
    value = 0
    while True:
        b = stream.read(1)
        if not b:
            break
        tmp = b[0]
        value = (value << 7) | (tmp & 0x7F)
        if (tmp & 0x80) == 0:
            break
    return value


def parse_svlq(stream):
    """Parse signed VLQ using zigzag-like mapping."""
    v = parse_vlq(stream)
    if (v & 1) == 0:
        return v >> 1
    return -((v >> 1) + 1)


def parse_starbytearray(stream):
    n = parse_vlq(stream)
    return stream.read(n)


def parse_starstring(stream):
    s = parse_starbytearray(stream)
    try:
        return str(s, encoding="utf-8")
    except UnicodeDecodeError:
        return s


def parse_variant_variant(stream):
    n = parse_vlq(stream)
    return [parse_variant(stream) for _ in range(n)]


def parse_dict_variant(stream):
    n = parse_vlq(stream)
    out = {}
    for _ in range(n):
        key = parse_starstring(stream)
        value = parse_variant(stream)
        out[key] = value
    return out


def parse_variant(stream):
    """Parse tagged variant value from a byte stream."""
    b = stream.read(1)
    if not b:
        return None
    tag = b[0]

    if tag == 1:
        return None
    if tag == 2:
        return struct.unpack(">d", stream.read(8))[0]
    if tag == 3:
        c = stream.read(1)
        return c == b"\x01"
    if tag == 4:
        return parse_svlq(stream)
    if tag == 5:
        return parse_starstring(stream)
    if tag == 6:
        return parse_variant_variant(stream)
    if tag == 7:
        return parse_dict_variant(stream)
    return None
