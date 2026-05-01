from __future__ import annotations


def run_lengths(bits: list[int]) -> list[tuple[int, int]]:
    """Return (bit, run_len) tuples for a 0/1 sequence."""
    if not bits:
        return []
    out: list[tuple[int, int]] = []
    cur = bits[0]
    n = 1
    for b in bits[1:]:
        if b == cur:
            n += 1
        else:
            out.append((cur, n))
            cur = b
            n = 1
    out.append((cur, n))
    return out


def normalize_runs(runs: list[tuple[int, int]]) -> list[tuple[int, float]]:
    tot = sum(n for _, n in runs)
    if tot == 0:
        return [(b, 0.0) for b, _ in runs]
    return [(b, n / tot) for b, n in runs]
