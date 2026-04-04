"""Accumulate checksum from mixed range-loop patterns.

Keywords: algorithms, loop optimization, range iteration, checksum, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(12, 180, 3, 2, 90000))
def range_loop_checksum(a: int, b: int, step: int, factor: int, rounds: int) -> tuple:
    """Combine several range loop forms and return deterministic loop metrics."""
    checksum = 0
    last_i = 0

    for r in range(rounds):
        for i in range(a):
            checksum += i + (r & 3)
            last_i = i

        for i in range(a, b):
            checksum += (i * factor) ^ (r & 7)
            last_i = i

        i = 0
        while i < b:
            checksum += i * (step + 1)
            i += step
        last_i = i

    return (checksum & 0xFFFFFFFF, last_i, rounds)
