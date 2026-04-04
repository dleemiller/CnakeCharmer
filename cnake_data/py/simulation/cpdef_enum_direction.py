"""Random walk using cardinal directions, return final Manhattan distance.

Keywords: simulation, random walk, enum, direction, benchmark
"""

from cnake_data.benchmarks import python_benchmark

NORTH = 0
EAST = 1
SOUTH = 2
WEST = 3


@python_benchmark(args=(100000,))
def cpdef_enum_direction(n: int) -> int:
    """Simulate a random walk using deterministic direction choices.

    Direction at step i = ((i * 2654435761) >> 16) % 4.
    NORTH: y+1, EAST: x+1, SOUTH: y-1, WEST: x-1.

    Args:
        n: Number of steps.

    Returns:
        Final Manhattan distance |x| + |y|.
    """
    x = 0
    y = 0
    for i in range(n):
        direction = ((i * 2654435761) >> 16) % 4
        if direction == NORTH:
            y += 1
        elif direction == EAST:
            x += 1
        elif direction == SOUTH:
            y -= 1
        else:
            x -= 1

    return abs(x) + abs(y)
