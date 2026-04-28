"""C3 method-resolution-order merge utility."""

from __future__ import annotations


def c3_merge(sequences: list[list[object]]) -> list[object]:
    seqs = [list(s) for s in sequences if s]
    out: list[object] = []

    while seqs:
        candidate = None
        for seq in seqs:
            head = seq[0]
            if not any(head in other[1:] for other in seqs):
                candidate = head
                break
        if candidate is None:
            tails = [repr(s[0]) for s in seqs if s]
            raise ValueError(f"Can not merge the items {', '.join(tails)}.")

        out.append(candidate)
        next_seqs = []
        for seq in seqs:
            if seq and seq[0] is candidate:
                seq.pop(0)
            if seq:
                next_seqs.append(seq)
        seqs = next_seqs

    return out


def c3_algorithm(
    start: object, bases_attr: str, linearized_attr: str, proper: bool
) -> list[object]:
    bases = list(getattr(start, bases_attr))
    merged = c3_merge([list(getattr(base, linearized_attr)) for base in bases])
    return merged if proper else [start] + merged
