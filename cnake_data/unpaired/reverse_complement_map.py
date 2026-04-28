BASEMAP = {
    "A": "T",
    "C": "G",
    "G": "C",
    "T": "A",
    "N": "N",
    "a": "t",
    "c": "g",
    "g": "c",
    "t": "a",
    "n": "n",
}


def reverse_complement(seq):
    """Return reverse-complement DNA sequence with passthrough for unknown chars."""
    out = [""] * len(seq)
    n = len(seq)
    for i, ch in enumerate(seq):
        out[n - i - 1] = BASEMAP.get(ch, ch)
    return "".join(out)
