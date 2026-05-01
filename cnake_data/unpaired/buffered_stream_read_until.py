from __future__ import annotations


def gen_read_until(chunks: list[bytes], delim: bytes) -> list[bytes]:
    out: list[bytes] = []
    buf = b""
    m = 0
    ld = len(delim)
    idx = 0
    while True:
        if not buf:
            if idx >= len(chunks):
                out.append(b"")
                return out
            buf = chunks[idx]
            idx += 1
        i = 0
        lb = len(buf)
        while i < lb:
            if buf[i] == delim[m]:
                m += 1
                if m == ld:
                    out.append(buf[: i + 1])
                    return out
            else:
                m = 0
            i += 1
        out.append(buf)
        buf = b""
