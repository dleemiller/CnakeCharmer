"""Thread-oriented buffered read skeleton."""

from __future__ import annotations


def run(fd_readers, buf_size=1000000, read_size=50000):
    n_bufs = len(fd_readers)
    offsets = [i * buf_size for i in range(n_bufs)]
    gots = [0] * n_bufs
    dones = [0] * n_bufs
    total_gots = [0] * n_bufs

    for idx, reader in enumerate(fd_readers):
        while dones[idx] == 0:
            if offsets[idx] + read_size > buf_size * (idx + 1):
                offsets[idx] = 0
            data = reader(read_size)
            gots[idx] = len(data)
            total_gots[idx] += gots[idx]
            if gots[idx] <= 0:
                dones[idx] = 1
            else:
                offsets[idx] += gots[idx]

    return total_gots
