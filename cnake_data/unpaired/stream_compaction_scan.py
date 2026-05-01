from __future__ import annotations


def exclusive_scan(flags: list[int]) -> list[int]:
    out = [0] * len(flags)
    s = 0
    for i, f in enumerate(flags):
        out[i] = s
        s += 1 if f else 0
    return out


def compact(values: list[float], flags: list[int]) -> list[float]:
    if len(values) != len(flags):
        raise ValueError("length mismatch")
    pos = exclusive_scan(flags)
    n_out = pos[-1] + (1 if flags and flags[-1] else 0)
    out = [0.0] * n_out
    for i, v in enumerate(values):
        if flags[i]:
            out[pos[i]] = v
    return out
