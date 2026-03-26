"""Schelling segregation model on a 1D ring with deterministic moves.

Keywords: simulation, Schelling, segregation, agent-based, 1D ring, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(5000,))
def schelling_segregation(n: int) -> tuple:
    """Simulate Schelling segregation on a 1D ring of n cells.

    Cells are initialized deterministically: cell[i] = i % 3 (0=empty, 1=type A,
    2=type B). Each agent is unhappy if fewer than threshold fraction of its
    non-empty neighbors (within radius 3) are the same type. Unhappy agents move
    to the nearest empty cell to the right (wrapping). Runs for 200 iterations.

    Args:
        n: Size of the ring.

    Returns:
        Tuple of (count_type_a, count_type_b, num_happy, total_moves).
    """
    threshold = 0.5
    radius = 3
    iterations = 200

    # Initialize ring
    ring = [0] * n
    for i in range(n):
        ring[i] = i % 3

    total_moves = 0

    for _ in range(iterations):
        # Find empty positions
        empties = [0] * n
        n_empty = 0
        for i in range(n):
            if ring[i] == 0:
                empties[n_empty] = i
                n_empty += 1

        if n_empty == 0:
            break

        # Check happiness and move unhappy agents
        moved = [0] * n  # track if cell was already moved this iteration

        for i in range(n):
            if ring[i] == 0 or moved[i]:
                continue

            agent_type = ring[i]
            same = 0
            total_neighbors = 0

            for d in range(-radius, radius + 1):
                if d == 0:
                    continue
                j = (i + d) % n
                if ring[j] != 0:
                    total_neighbors += 1
                    if ring[j] == agent_type:
                        same += 1

            # Check if unhappy
            if total_neighbors > 0 and same / total_neighbors < threshold:
                # Find nearest empty to the right
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
            j = (i + d) % n
            if ring[j] != 0:
                total_neighbors += 1
                if ring[j] == agent_type:
                    same += 1

        if total_neighbors == 0 or same / total_neighbors >= threshold:
            num_happy += 1

    return (count_a, count_b, num_happy, total_moves)
