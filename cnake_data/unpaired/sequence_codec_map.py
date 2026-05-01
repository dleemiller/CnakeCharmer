from __future__ import annotations


def encode_chars(alphabet: bytes, symbols: bytes) -> list[int]:
    illegal = len(alphabet)
    sym_to_code = [illegal] * 256
    for i, b in enumerate(alphabet):
        sym_to_code[b] = i
    out: list[int] = [0] * len(symbols)
    for i, b in enumerate(symbols):
        code = sym_to_code[b]
        if code == illegal:
            raise ValueError("symbol " + repr(chr(b)) + " not in alphabet")
        out[i] = code
    return out


def decode_to_chars(alphabet: bytes, code: list[int]) -> bytes:
    out = bytearray(len(code))
    n = len(alphabet)
    for i, c in enumerate(code):
        if c >= n:
            raise ValueError(f"invalid code {c}")
        out[i] = alphabet[c]
    return bytes(out)


def map_sequence_code(mapping: list[int], in_code: list[int]) -> list[int]:
    out = [0] * len(in_code)
    for i, c in enumerate(in_code):
        out[i] = mapping[c]
    return out
