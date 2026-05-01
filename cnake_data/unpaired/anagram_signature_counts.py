from __future__ import annotations


def signature(word: str) -> str:
    counts = [0] * 26
    for ch in word.lower():
        o = ord(ch) - 97
        if 0 <= o < 26:
            counts[o] += 1
    return ":".join(str(c) for c in counts)


def group_anagrams(words: list[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for w in words:
        s = signature(w)
        out.setdefault(s, []).append(w)
    return out
