"""Sort 8-byte records and count unique via comparison.

Keywords: algorithms, memcmp, dedup, sort, unique, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(50000,))
def memcmp_dedup_count(n: int) -> int:
    """Generate n 8-int records, sort, count unique.

    Args:
        n: Number of records.

    Returns:
        Count of unique records.
    """
    rec_len = 8
    records = []
    for i in range(n):
        rec = [0] * rec_len
        seed = i * 2654435761 + 17
        for j in range(rec_len):
            seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
            rec[j] = seed % 256
        records.append(tuple(rec))

    records.sort()

    unique = 1
    for i in range(1, n):
        if records[i] != records[i - 1]:
            unique += 1

    return unique
