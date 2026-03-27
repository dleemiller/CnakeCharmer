"""Hash table simulation using enum-defined constants.

Keywords: numerical, enum, hash table, constants, simulation, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark

MAX_SIZE = 1024
BUCKET_COUNT = 256
HASH_SEED = 0x9E3779B9


@python_benchmark(args=(100000,))
def anon_enum_limits(n: int) -> int:
    """Simulate a hash table using enum-defined constants.

    Inserts n values into a fixed-size hash table with linear probing,
    then counts total collisions.

    Args:
        n: Number of insert operations.

    Returns:
        Total number of collisions during insertion.
    """
    table = [0] * MAX_SIZE
    occupied = [0] * MAX_SIZE
    collisions = 0

    for i in range(n):
        # Hash the value using HASH_SEED
        val = ((i * HASH_SEED) ^ (i >> 5)) & 0xFFFFFFFF
        bucket = val % BUCKET_COUNT

        # Linear probe within the bucket's region
        region_start = (bucket * (MAX_SIZE // BUCKET_COUNT)) % MAX_SIZE
        placed = 0
        for step in range(MAX_SIZE // BUCKET_COUNT):
            slot = (region_start + step) % MAX_SIZE
            if occupied[slot] == 0:
                table[slot] = val
                occupied[slot] = 1
                placed = 1
                break
            else:
                collisions += 1

        if placed == 0:
            # Overflow: wrap around entire table
            for slot in range(MAX_SIZE):
                if occupied[slot] == 0:
                    table[slot] = val
                    occupied[slot] = 1
                    break
                else:
                    collisions += 1

        # Periodically clear some entries to avoid saturation
        if i % (MAX_SIZE // 2) == 0 and i > 0:
            for j in range(MAX_SIZE):
                if j % 4 == 0:
                    occupied[j] = 0
                    table[j] = 0

    # Final checksum from table
    checksum = 0
    for j in range(MAX_SIZE):
        if occupied[j]:
            checksum ^= table[j]

    return collisions + (checksum & 0xFFFF)
