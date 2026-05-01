from __future__ import annotations


def evaluate_core(
    basis_row: list[float],
    core_params: list[float],
    core_params_map: list[int],
    ranks0: int,
    ranks1: int,
) -> list[list[float]]:
    out = [[0.0 for _ in range(ranks1)] for _ in range(ranks0)]
    num_core_params = len(core_params)
    num_funcs = len(core_params_map) - 1

    for j in range(ranks1):
        for i in range(ranks0):
            fn = j * ranks0 + i
            lb = core_params_map[fn]
            ub = num_core_params if fn == num_funcs else core_params_map[fn + 1]
            s = 0.0
            for n in range(ub - lb):
                s += basis_row[n] * core_params[lb + n]
            out[i][j] = s
    return out
