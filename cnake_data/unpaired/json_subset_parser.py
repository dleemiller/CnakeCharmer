from __future__ import annotations


def parse_int_array(text: str) -> list[int]:
    """Parse a strict JSON subset: arrays of signed integers, e.g. [1,-2,3]."""
    i = 0
    n = len(text)
    while i < n and text[i].isspace():
        i += 1
    if i >= n or text[i] != "[":
        raise ValueError("expected '['")
    i += 1

    out: list[int] = []
    need_value = True
    while i < n:
        while i < n and text[i].isspace():
            i += 1
        if i < n and text[i] == "]":
            if need_value and out:
                raise ValueError("trailing comma")
            i += 1
            break

        if not need_value:
            if i >= n or text[i] != ",":
                raise ValueError("expected ','")
            i += 1
            need_value = True
            continue

        sign = 1
        if i < n and text[i] in "+-":
            sign = -1 if text[i] == "-" else 1
            i += 1
        if i >= n or not text[i].isdigit():
            raise ValueError("expected digit")

        val = 0
        while i < n and text[i].isdigit():
            val = val * 10 + (ord(text[i]) - 48)
            i += 1
        out.append(sign * val)
        need_value = False

    while i < n and text[i].isspace():
        i += 1
    if i != n:
        raise ValueError("extra trailing content")
    return out
