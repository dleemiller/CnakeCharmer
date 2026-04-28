"""PEM codec and constant-time byte utilities."""

from __future__ import annotations

import base64
import re

_PEM_RE = re.compile(
    r"^\s*-----BEGIN\ (([A-Z]+\ )?(PUBLIC|PRIVATE))\ KEY-----(.+)-----END\ \1\ KEY-----\s*$",
    re.VERBOSE | re.DOTALL,
)


def pem_encode(type_: str, mode: str, content: bytes):
    type_ = type_.upper()
    mode = mode.upper()
    if mode not in ("PUBLIC", "PRIVATE"):
        raise ValueError("mode must be PUBLIC or PRIVATE")
    t = f"{type_} {mode}" if mode == "PRIVATE" else "PUBLIC"
    b64 = base64.b64encode(content).decode()
    b64 = "\n".join(b64[i : i + 78] for i in range(0, len(b64), 78))
    return f"-----BEGIN {t} KEY-----\n{b64}\n-----END {t} KEY-----\n"


def pem_decode(content: str):
    m = _PEM_RE.match(content)
    if not m:
        raise ValueError("not PEM encoded")
    type_, mode, body = m.group(2, 3, 4)
    type_ = type_ and type_.rstrip().upper()
    mode = mode.upper()
    data = base64.b64decode("".join(body.strip().split()).encode())
    return type_, mode, data


def xor_bytes(a: bytes, b: bytes):
    if len(a) != len(b):
        raise ValueError(f"arguments must have matching lengths; given {len(a)} and {len(b)}")
    return bytes(x ^ y for x, y in zip(a, b, strict=False))


def bytes_equal(a: bytes, b: bytes):
    if len(a) != len(b):
        raise ValueError(f"arguments must have matching lengths; given {len(a)} and {len(b)}")
    are_different = 0
    for x, y in zip(a, b, strict=False):
        are_different |= x ^ y
    return not are_different
