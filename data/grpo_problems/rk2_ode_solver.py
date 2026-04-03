def rk2_ode_solver(n_steps):
    """Solve dy/dt = -y using RK2 (midpoint method) from t=0 to t=5.

    Initial condition y(0) = 1.0. The exact solution is exp(-t).
    Returns (final_y, final_t, max_error) over the trajectory.

    Args:
        n_steps: Number of integration steps.
    """
    import math

    t = 0.0
    y = 1.0
    dt = 5.0 / n_steps
    max_error = 0.0

    for _ in range(n_steps):
        # RK2 midpoint
        k1 = -y
        y_mid = y + 0.5 * dt * k1
        k2 = -y_mid
        y = y + dt * k2
        t += dt

        exact = math.exp(-t)
        error = abs(y - exact)
        if error > max_error:
            max_error = error

    return (round(y, 10), round(t, 10), round(max_error, 10))
