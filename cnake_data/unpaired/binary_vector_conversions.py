from __future__ import annotations


def dense_to_binary_char(np_array: list[int]) -> list[int]:
    out = [0] * len(np_array)
    for i in range(len(np_array)):
        out[i] = 1 if np_array[i] else 0
    return out


def sparse_cols_to_binary(n_cols: int, nonzero_coords: list[tuple[int, int]]) -> list[int]:
    out = [0] * n_cols
    for _, j in nonzero_coords:
        if 0 <= j < n_cols:
            out[j] = 1
    return out


def binary_to_float(vec: list[int]) -> list[float]:
    out = [0.0] * len(vec)
    for i in range(len(vec)):
        out[i] = float(vec[i])
    return out
