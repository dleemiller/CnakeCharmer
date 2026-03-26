# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Schelling segregation model on a 1D ring with deterministic moves (Cython-optimized).

Keywords: simulation, Schelling, segregation, agent-based, 1D ring, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from libc.string cimport memset
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(5000,))
def schelling_segregation(int n):
    """Simulate Schelling segregation on a 1D ring of n cells."""
    cdef double threshold = 0.5
    cdef int radius = 3
    cdef int iterations = 200
    cdef int total_moves = 0
    cdef int i, d, j, it, search, pos, best_pos
    cdef int agent_type, same, total_neighbors
    cdef int count_a, count_b, num_happy

    cdef int *ring = <int *>malloc(n * sizeof(int))
    cdef int *moved = <int *>malloc(n * sizeof(int))
    if not ring or not moved:
        if ring: free(ring)
        if moved: free(moved)
        raise MemoryError()

    for i in range(n):
        ring[i] = i % 3

    for it in range(iterations):
        memset(moved, 0, n * sizeof(int))

        for i in range(n):
            if ring[i] == 0 or moved[i]:
                continue

            agent_type = ring[i]
            same = 0
            total_neighbors = 0

            for d in range(-radius, radius + 1):
                if d == 0:
                    continue
                j = (i + d + n) % n
                if ring[j] != 0:
                    total_neighbors += 1
                    if ring[j] == agent_type:
                        same += 1

            if total_neighbors > 0 and <double>same / total_neighbors < threshold:
                best_pos = -1
                for search in range(1, n):
                    pos = (i + search) % n
                    if ring[pos] == 0 and not moved[pos]:
                        best_pos = pos
                        break

                if best_pos >= 0:
                    ring[best_pos] = agent_type
                    ring[i] = 0
                    moved[best_pos] = 1
                    moved[i] = 1
                    total_moves += 1

    # Count final statistics
    count_a = 0
    count_b = 0
    num_happy = 0

    for i in range(n):
        if ring[i] == 1:
            count_a += 1
        elif ring[i] == 2:
            count_b += 1

        if ring[i] == 0:
            continue

        agent_type = ring[i]
        same = 0
        total_neighbors = 0
        for d in range(-radius, radius + 1):
            if d == 0:
                continue
            j = (i + d + n) % n
            if ring[j] != 0:
                total_neighbors += 1
                if ring[j] == agent_type:
                    same += 1

        if total_neighbors == 0 or <double>same / total_neighbors >= threshold:
            num_happy += 1

    free(ring)
    free(moved)
    return (count_a, count_b, num_happy, total_moves)
