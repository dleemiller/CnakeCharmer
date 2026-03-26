"""Linear-probing hash table insert and lookup operations.

Keywords: algorithms, hash table, linear probing, lookup, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(500000,))
def hash_table_ops(n: int) -> int:
    """Insert n keys into a linear-probing hash table, count query hits.

    Table size: 2*n. Keys: (i*31+17) % 100000. Queries: (i*37+13) % 100000.

    Args:
        n: Number of keys to insert and queries to perform.

    Returns:
        Number of queries that found a matching key.
    """
    table_size = 2 * n
    EMPTY = -1
    table = [EMPTY] * table_size

    # Insert keys
    for i in range(n):
        key = (i * 31 + 17) % 100000
        slot = key % table_size
        while table[slot] != EMPTY:
            if table[slot] == key:
                break
            slot = (slot + 1) % table_size
        table[slot] = key

    # Count query hits
    hits = 0
    for i in range(n):
        query = (i * 37 + 13) % 100000
        slot = query % table_size
        while table[slot] != EMPTY:
            if table[slot] == query:
                hits += 1
                break
            slot = (slot + 1) % table_size

    return hits
