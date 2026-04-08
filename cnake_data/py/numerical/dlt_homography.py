"""Reprojection error evaluation for homography correspondences.

Computes symmetric reprojection errors for point correspondences
under a given homography model — a core step in DLT-based calibration.

Keywords: DLT, reprojection error, homography, camera calibration, geometric vision
"""

import math

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(1000,))
def dlt_homography(n_pts: int) -> tuple:
    """Evaluate reprojection errors for noisy homography correspondences.

    Generates n_pts spiral correspondences under a known homography H_gt,
    adds Gaussian-like noise, then evaluates their errors under a perturbed
    model H_est. Accumulates error statistics.

    Args:
        n_pts: Number of point correspondences.

    Returns:
        Tuple of (sum_error, max_error, n_inliers) using threshold=2.0 px.
    """
    # Ground-truth homography
    h00 = 1.05
    h01 = 0.02
    h02 = 8.0
    h10 = -0.01
    h11 = 0.98
    h12 = 5.0
    h20 = 0.0001
    h21 = 0.00005
    h22 = 1.0

    # Perturbed model (estimated homography with small error)
    e00 = h00 + 0.01
    e01 = h01
    e02 = h02 + 1.2
    e10 = h10
    e11 = h11 - 0.005
    e12 = h12 + 0.8
    e20 = h20
    e21 = h21
    e22 = h22

    sum_err = 0.0
    max_err = 0.0
    n_inliers = 0
    threshold = 2.0

    for i in range(n_pts):
        t = 2.0 * math.pi * i / n_pts
        sx = 50.0 + 40.0 * math.cos(t)
        sy = 50.0 + 40.0 * math.sin(t)

        # True destination under H_gt
        w_gt = h20 * sx + h21 * sy + h22
        dx_gt = (h00 * sx + h01 * sy + h02) / w_gt
        dy_gt = (h10 * sx + h11 * sy + h12) / w_gt

        # Noisy destination (deterministic noise via sin)
        noise_x = 0.3 * math.sin(i * 1.7)
        noise_y = 0.3 * math.cos(i * 2.3)
        dx_obs = dx_gt + noise_x
        dy_obs = dy_gt + noise_y

        # Predicted destination under H_est
        w_est = e20 * sx + e21 * sy + e22
        dx_pred = (e00 * sx + e01 * sy + e02) / w_est
        dy_pred = (e10 * sx + e11 * sy + e12) / w_est

        ex = dx_pred - dx_obs
        ey = dy_pred - dy_obs
        err = math.sqrt(ex * ex + ey * ey)
        sum_err += err
        if err > max_err:
            max_err = err
        if err < threshold:
            n_inliers += 1

    return (sum_err, max_err, n_inliers)
