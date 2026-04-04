"""Hash table with typedef'd key type, insert n values and count collisions.

Keywords: algorithms, hash table, typedef, collision, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(50000,))
def typedef_hash_table(n: int) -> int:
    """Insert n values into a hash table and count collisions.

    Hash function: key = (i * 2654435761) % (2**32), bucket = key % table_size.
    Table size = n * 2. A collision occurs when a bucket is already occupied.

    Args:
        n: Number of values to insert.

    Returns:
        Number of collisions.
    """
    table_size = n * 2
    occupied = [0] * table_size
    collisions = 0

    mask = 0xFFFFFFFF
    for i in range(n):
        key = (i * 2654435761) & mask
        bucket = key % table_size
        if occupied[bucket]:
            collisions += 1
        occupied[bucket] = 1

    return collisions
