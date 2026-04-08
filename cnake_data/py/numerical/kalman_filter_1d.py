"""1D Kalman filter for scalar state estimation.

Implements a standard discrete-time Kalman filter for a 1D constant-velocity
model. Processes a noisy measurement sequence and returns estimation statistics.

Keywords: Kalman filter, state estimation, control, temporal filtering, Bayesian
"""

import math

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(5000,))
def kalman_filter_1d(n_steps: int) -> tuple:
    """Run a 1D Kalman filter on a deterministic noisy measurement sequence.

    State: position x. Transition: x_{k+1} = x_k + v_k (constant velocity).
    Measurement: z_k = x_k + noise_k.

    Args:
        n_steps: Number of filter steps.

    Returns:
        Tuple of (sum_x_est, sum_innov, final_gain) — sum of state estimates,
        sum of innovations, and the final Kalman gain.
    """
    # Filter parameters
    Q = 0.01  # process noise variance
    R = 1.0  # measurement noise variance
    A = 1.0  # state transition (identity for constant model)
    H = 1.0  # observation model

    # Initial state
    x_est = 0.0
    P = 1.0  # initial error covariance

    sum_x = 0.0
    sum_innov = 0.0
    gain = 0.0

    # Deterministic "noisy" measurements: sin wave + LCG noise
    lcg = 12345
    lcg_a = 1664525
    lcg_c = 1013904223
    lcg_m = 2**32

    for k in range(n_steps):
        # Generate measurement
        true_x = math.sin(k * 0.02)
        lcg = (lcg_a * lcg + lcg_c) % lcg_m
        noise = (lcg / lcg_m - 0.5) * 2.0  # [-1, 1]
        z = true_x + noise

        # Predict
        x_pred = A * x_est
        P_pred = A * P * A + Q

        # Update
        S = H * P_pred * H + R
        gain = P_pred * H / S
        innov = z - H * x_pred
        x_est = x_pred + gain * innov
        P = (1.0 - gain * H) * P_pred

        sum_x += x_est
        sum_innov += innov

    return (sum_x, sum_innov, gain)
