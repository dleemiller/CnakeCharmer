import math


def wave_propagation(
    num_steps,
    scale=100,
    damping=0.25,
    initial_pressure=250.0,
    stop_step=100,
):
    """Simulate damped 2D wave pressure propagation on a grid.

    Returns final pressure grid as nested lists.
    """
    omega = 3.0 / (2.0 * math.pi)

    size_x = 2 * scale + 1
    size_y = 2 * scale + 1
    center_y = scale
    center_x = scale

    pressure = [[0.0 for _ in range(size_y)] for _ in range(size_x)]
    velocity = [[[0.0, 0.0, 0.0, 0.0] for _ in range(size_y)] for _ in range(size_x)]
    pressure[center_y][center_x] = initial_pressure

    for step in range(num_steps):
        if step <= stop_step:
            pressure[center_y][center_x] = initial_pressure * math.sin(omega * step)

        for i in range(size_x):
            for j in range(size_y):
                p = pressure[i][j]
                velocity[i][j][0] = velocity[i][j][0] + (p - pressure[i - 1][j] if i > 0 else p)
                velocity[i][j][1] = velocity[i][j][1] + (
                    p - pressure[i][j + 1] if j < size_y - 1 else p
                )
                velocity[i][j][2] = velocity[i][j][2] + (
                    p - pressure[i + 1][j] if i < size_x - 1 else p
                )
                velocity[i][j][3] = velocity[i][j][3] + (p - pressure[i][j - 1] if j > 0 else p)

        for i in range(size_x):
            for j in range(size_y):
                pressure[i][j] -= 0.5 * damping * sum(velocity[i][j])

    return pressure
