from __future__ import annotations


def reverse_complement(seq: str) -> str:
    table = {
        "A": "T",
        "C": "G",
        "G": "C",
        "T": "A",
        "a": "t",
        "c": "g",
        "g": "c",
        "t": "a",
        "N": "N",
        "n": "n",
    }
    out = ["N"] * len(seq)
    n = len(seq)
    for i, ch in enumerate(seq):
        out[n - 1 - i] = table.get(ch, "N")
    return "".join(out)
