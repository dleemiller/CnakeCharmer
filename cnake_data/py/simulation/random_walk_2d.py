"""
2D random walk with n steps using deterministic LCG for direction.

Keywords: simulation, random walk, 2d, LCG, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(5000000,))
def random_walk_2d(n: int) -> tuple:
    """2D random walk: n steps, 4 directions, deterministic LCG.

    Direction for step i: dir = (i * 1664525 + 1013904223) & 3
      0: x+=1, 1: y+=1, 2: x-=1, 3: y-=1

    Args:
        n: Number of steps.

    Returns:
        Tuple of (x, y, max_x_ever, min_y_ever, steps_at_origin).
    """
    x = 0
    y = 0
    max_x = 0
    min_y = 0
    steps_at_origin = 0

    for i in range(n):
        direction = (i * 1664525 + 1013904223) & 3
        if direction == 0:
            x += 1
        elif direction == 1:
            y += 1
        elif direction == 2:
            x -= 1
        else:
            y -= 1

        if x > max_x:
            max_x = x
        if y < min_y:
            min_y = y
        if x == 0 and y == 0:
            steps_at_origin += 1

    return (x, y, max_x, min_y, steps_at_origin)
