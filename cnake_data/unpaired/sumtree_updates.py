from __future__ import annotations


def build_sumtree(values: list[float]) -> list[float]:
    n = len(values)
    size = 1
    while size < n:
        size *= 2
    tree = [0.0] * (2 * size)
    for i, v in enumerate(values):
        tree[size + i] = v
    for i in range(size - 1, 0, -1):
        tree[i] = tree[2 * i] + tree[2 * i + 1]
    return tree


def update_sumtree(tree: list[float], idx: int, value: float) -> None:
    n2 = len(tree)
    size = n2 // 2
    i = size + idx
    tree[i] = value
    i //= 2
    while i >= 1:
        tree[i] = tree[2 * i] + tree[2 * i + 1]
        i //= 2


def sample_prefix(tree: list[float], target: float) -> int:
    i = 1
    size = len(tree) // 2
    while i < size:
        left = 2 * i
        if target <= tree[left]:
            i = left
        else:
            target -= tree[left]
            i = left + 1
    return i - size
