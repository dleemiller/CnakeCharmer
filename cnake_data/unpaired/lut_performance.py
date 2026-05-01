"""Dictionary lookup vs computed-index micro-benchmark helpers."""

from __future__ import annotations

from collections import OrderedDict
from datetime import datetime

import numpy as np


def lookup_0(d, block_positions):
    total = 0
    for p in block_positions:
        total += d[p]
    return total


def calc_0(d, block_positions):
    total = 0
    for p in block_positions:
        total += p[0] * 34 + p[1]
    return total


def my_timeit(fname: str, d, block_positions, repeat: int, run_count: int):
    runtimes = np.empty(repeat, dtype=float)
    for r in range(repeat):
        start = datetime.now()
        for _ in range(run_count):
            if fname == "calc_0":
                calc_0(d, block_positions)
            elif fname == "lookup_0":
                lookup_0(d, block_positions)
            else:
                raise ValueError(f"Unknown f: {fname}")
        end = datetime.now()
        runtimes[r] = (end - start).total_seconds()
    return float(runtimes.min())


def run_time_test(run_count: int = 10000, load_factor: int = 1):
    runtimes = OrderedDict()
    d = dict(zip([(x, y) for x in range(34) for y in range(34)], range(34 * 34), strict=False))
    positions = list(d.keys())
    block_positions = positions[:]
    rand_state = np.random.RandomState()
    rand_state.shuffle(block_positions)

    runtimes["lut"] = my_timeit("lookup_0", d, block_positions, 3, load_factor * run_count)
    runtimes["calc"] = my_timeit("calc_0", d, block_positions, 3, load_factor * run_count)
    return runtimes
