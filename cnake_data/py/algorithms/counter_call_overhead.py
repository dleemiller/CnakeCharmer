"""
Compare direct accumulation vs helper-call and generator-call styles.

Sourced from SFT DuckDB blob: dd6f108767d588dd0632ce9cb0d07b1da90c822a
Keywords: loops, helper function, generator, call overhead, algorithms
"""

from cnake_data.benchmarks import python_benchmark


def _helper(v: int) -> int:
    return v


def _helper_gen(limit: int, offset: int):
    for i in range(limit):
        yield i + offset


@python_benchmark(args=(200000, 4, 3))
def counter_call_overhead(limit: int, repeats: int, offset: int) -> tuple:
    """Return sums for direct, helper-call, and generator-call strategies."""
    direct = 0
    via_helper = 0
    via_gen = 0

    for _ in range(repeats):
        for i in range(limit):
            direct += i + offset
            via_helper += _helper(i + offset)
        for v in _helper_gen(limit, offset):
            via_gen += v

    return (direct, via_helper, via_gen)
