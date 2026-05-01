from __future__ import annotations


def prefix_offsets(lengths: list[int]) -> list[int]:
    out = [0] * (len(lengths) + 1)
    s = 0
    for i, n in enumerate(lengths):
        s += n
        out[i + 1] = s
    return out


def locate(offsets: list[int], flat_index: int) -> tuple[int, int]:
    """Map a flat index into (row, col) using monotone prefix offsets."""
    if flat_index < 0 or flat_index >= offsets[-1]:
        raise IndexError("flat index out of bounds")
    lo, hi = 0, len(offsets) - 2
    while lo <= hi:
        mid = (lo + hi) // 2
        if offsets[mid] <= flat_index < offsets[mid + 1]:
            return mid, flat_index - offsets[mid]
        if flat_index < offsets[mid]:
            hi = mid - 1
        else:
            lo = mid + 1
    raise RuntimeError("unreachable")
