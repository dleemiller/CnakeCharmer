"""Sort dataclass-like records by key and compute checksum.

Keywords: algorithms, dataclass, sorting, record, extension type, benchmark
"""

from cnake_data.benchmarks import python_benchmark


class Record:
    """Simple record with key and value."""

    __slots__ = ("key", "value")

    def __init__(self, key, value):
        self.key = key
        self.value = value


@python_benchmark(args=(50000,))
def dataclass_record_sort(n: int) -> int:
    """Create n records, sort by key, return checksum.

    Args:
        n: Number of records to create and sort.

    Returns:
        Integer checksum from sorted record traversal.
    """
    records = [None] * n
    for i in range(n):
        key = ((i * 2654435761) ^ (i >> 3)) & 0x7FFFFFFF
        value = ((i * 1664525 + 1013904223) % 100000) / 100.0
        records[i] = Record(key, value)

    records.sort(key=lambda r: r.key)

    checksum = 0
    for i in range(n):
        checksum = (checksum * 31 + records[i].key) & 0xFFFFFFFF
    return checksum
