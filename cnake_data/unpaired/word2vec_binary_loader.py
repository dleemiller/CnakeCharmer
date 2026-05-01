from __future__ import annotations


def _read_token(data: bytes, i: int) -> tuple[str, int]:
    j = i
    n = len(data)
    while j < n and data[j] not in (32, 10):
        j += 1
    if j == i:
        raise ValueError("empty token")
    tok = data[i:j].decode("utf-8", errors="strict")
    while j < n and data[j] in (32, 10):
        j += 1
    return tok, j


def read_word2vec_header(data: bytes) -> tuple[int, int, int]:
    """Read (vocab_size, dim, offset_after_header) from binary word2vec blob."""
    vocab_s, i = _read_token(data, 0)
    dim_s, i = _read_token(data, i)
    return int(vocab_s), int(dim_s), i


def parse_float32_row(row: bytes) -> list[float]:
    if len(row) % 4 != 0:
        raise ValueError("row length must be multiple of 4")
    out: list[float] = []
    import struct

    for i in range(0, len(row), 4):
        out.append(struct.unpack_from("<f", row, i)[0])
    return out
