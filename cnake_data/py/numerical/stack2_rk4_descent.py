"""Trace a 2D RK4 descent path in a linear gradient field.

Adapted from The Stack v2 Cython candidate:
- blob_id: 731b31ede4e99a6989cdd0d6add9b508e2c754e1
- filename: optimal_path_cython.pyx

Keywords: numerical, rk4, gradient descent, trajectory, integration
"""

from cnake_data.benchmarks import python_benchmark


def _grad_x(coord_x: float, coord_y: float) -> float:
    return 2.0 * coord_x + 0.25 * coord_y


def _grad_y(coord_x: float, coord_y: float) -> float:
    return 2.0 * coord_y - 0.125 * coord_x


@python_benchmark(args=(1250, -980, 220000, 17, 200000))
def stack2_rk4_descent(
    start_x_milli: int,
    start_y_milli: int,
    step_count: int,
    step_milli: int,
    min_steps: int = 0,
) -> tuple:
    """Integrate downhill dynamics and return quantized path stats."""
    coord_x = start_x_milli / 1000.0
    coord_y = start_y_milli / 1000.0
    delta = step_milli / 1000.0

    travel = 0.0
    stop_iter = step_count
    for idx in range(step_count):
        k1x = -_grad_x(coord_x, coord_y)
        k1y = -_grad_y(coord_x, coord_y)
        k2x = -_grad_x(coord_x + 0.5 * delta * k1x, coord_y + 0.5 * delta * k1y)
        k2y = -_grad_y(coord_x + 0.5 * delta * k1x, coord_y + 0.5 * delta * k1y)
        k3x = -_grad_x(coord_x + 0.5 * delta * k2x, coord_y + 0.5 * delta * k2y)
        k3y = -_grad_y(coord_x + 0.5 * delta * k2x, coord_y + 0.5 * delta * k2y)
        k4x = -_grad_x(coord_x + delta * k3x, coord_y + delta * k3y)
        k4y = -_grad_y(coord_x + delta * k3x, coord_y + delta * k3y)

        next_x = coord_x + (delta / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
        next_y = coord_y + (delta / 6.0) * (k1y + 2.0 * k2y + 2.0 * k3y + k4y)

        dx = next_x - coord_x
        dy = next_y - coord_y
        travel += (dx * dx + dy * dy) ** 0.5

        coord_x = next_x
        coord_y = next_y

        if (idx + 1) >= min_steps and dx * dx + dy * dy < 1e-18:
            stop_iter = idx + 1
            break

    return (
        int(coord_x * 1_000_000),
        int(coord_y * 1_000_000),
        int(travel * 1_000_000) & 0xFFFFFFFF,
        stop_iter,
    )
