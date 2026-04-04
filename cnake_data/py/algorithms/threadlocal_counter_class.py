"""Class-based pseudo thread-local counters with weighted reduction.

Keywords: algorithms, class, thread-local, counters, benchmark
"""

from cnake_data.benchmarks import python_benchmark


class ThreadLocal:
    def __init__(self, nslots: int):
        self.counters = [0] * nslots

    def bump(self, slot: int, delta: int) -> int:
        self.counters[slot] += delta
        return self.counters[slot]


@python_benchmark(args=(64, 900000, 11, 5))
def threadlocal_counter_class(nslots: int, steps: int, seed: int, stride: int) -> tuple:
    tl = ThreadLocal(nslots)
    checksum = 0
    peak = 0
    for i in range(steps):
        slot = (seed + i * stride) % nslots
        delta = ((i * 13 + seed) % 7) - 3
        v = tl.bump(slot, delta)
        checksum += v * ((slot & 3) + 1)
        if v > peak:
            peak = v
    return (checksum, peak, tl.counters[nslots // 2])
