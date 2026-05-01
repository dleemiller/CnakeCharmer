from __future__ import annotations


def downsample_sum(array: list[list[float]], block_size: int) -> list[list[float]]:
    ny = len(array)
    nx = len(array[0]) if ny else 0
    nx_new = nx // block_size
    ny_new = ny // block_size
    out = [[0.0 for _ in range(nx_new)] for _ in range(ny_new)]
    for i in range(nx_new):
        for j in range(ny_new):
            s = 0.0
            for ii in range(block_size):
                for jj in range(block_size):
                    s += array[j * block_size + jj][i * block_size + ii]
            out[j][i] = s
    return out


def upsample_average(array: list[list[float]], block_size: int) -> list[list[float]]:
    ny = len(array)
    nx = len(array[0]) if ny else 0
    ny_new = ny * block_size
    nx_new = nx * block_size
    out = [[0.0 for _ in range(nx_new)] for _ in range(ny_new)]
    scale = float(block_size * block_size)
    for j in range(ny_new):
        src_j = j // block_size
        for i in range(nx_new):
            src_i = i // block_size
            out[j][i] = array[src_j][src_i] / scale
    return out
