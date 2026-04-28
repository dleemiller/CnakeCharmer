"""Lightweight text encoding utilities."""

from __future__ import annotations


def is_ascii(string: str) -> bool:
    for ch in string:
        if ord(ch) > 0x7F:
            return False
    return True


def encode_items_to_latin1(data: dict[str, str]) -> list[tuple[bytes, bytes]]:
    out: list[tuple[bytes, bytes]] = []
    for key, value in data.items():
        out.append((key.encode("latin1"), value.encode("latin1")))
    return out
