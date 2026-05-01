"""Map one list onto another via index lookups with optional tolerance to misses."""

from __future__ import annotations


def listmap(list1: list, list2: list, ignore_unmappable: bool = False) -> list[int]:
    idx = {v: i for i, v in enumerate(list2)}
    out: list[int] = []
    for v in list1:
        if v in idx:
            out.append(idx[v])
        elif ignore_unmappable:
            out.append(-1)
        else:
            raise KeyError(v)
    return out


def listmap_fill(list1: list, list2: list, fill_value: int = -1) -> list[int]:
    idx = {v: i for i, v in enumerate(list2)}
    return [idx.get(v, fill_value) for v in list1]
