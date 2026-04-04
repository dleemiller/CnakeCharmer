"""Tagged union pattern: process tagged int/double values and accumulate sum.

Keywords: union, tagged, struct, algorithms, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(100000,))
def union_tagged_value(n: int) -> float:
    """Process n tagged values (tag 0=int, tag 1=double), sum them.

    Uses a deterministic hash to assign tags and values,
    emulating a C tagged union pattern.

    Args:
        n: Number of tagged values to process.

    Returns:
        Accumulated sum of all tagged values as float.
    """
    total = 0.0
    for i in range(n):
        h = (i * 2654435761) & 0xFFFFFFFF
        tag = h % 2
        if tag == 0:
            # Integer value: use upper bits
            val = ((h >> 8) & 0xFFFF) - 32768
            total += val
        else:
            # Double value: scale hash to [-100.0, 100.0]
            val = ((h >> 8) & 0xFFFF) / 65535.0 * 200.0 - 100.0
            total += val
    return total
