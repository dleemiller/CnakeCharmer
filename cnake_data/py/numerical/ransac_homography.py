"""RANSAC robust homography inlier counting.

Iterates RANSAC trials, using an LCG to select random subsets of
correspondences, fits a model each trial, and accumulates inlier statistics.

Keywords: RANSAC, robust estimation, homography, inlier counting, geometric vision
"""

import math

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(200, 80))
def ransac_homography(n_pts: int, n_iter: int) -> tuple:
    """Run RANSAC inlier counting for a noisy homography correspondence set.

    Args:
        n_pts: Number of point correspondences.
        n_iter: Number of RANSAC iterations.

    Returns:
        Tuple of (best_inlier_count, total_checks, mean_reproj_error).
    """
    # Ground-truth homography
    h00 = 1.02
    h01 = 0.01
    h02 = 8.0
    h10 = -0.005
    h11 = 0.99
    h12 = 5.0
    h20 = 0.0002
    h21 = 0.00005
    h22 = 1.0

    # Generate correspondences with noise using deterministic sin/cos
    # (avoids LCG nonlocal issues; fully compatible with Cython)
    threshold2 = 4.0  # squared inlier threshold (2px)
    best_inliers = 0
    total_checks = 0
    total_err = 0.0

    for trial in range(n_iter):
        # Perturb homography slightly each trial (deterministic)
        scale = 1.0 + 0.005 * math.sin(trial * 1.3)
        offset = 0.5 * math.cos(trial * 0.7)

        e00 = h00 * scale
        e01 = h01
        e02 = h02 + offset
        e10 = h10
        e11 = h11 * scale
        e12 = h12 - offset * 0.5
        e20 = h20
        e21 = h21
        e22 = h22

        inliers = 0
        err_sum = 0.0

        for j in range(n_pts):
            # Deterministic source points on a grid
            sx = (j % 20) * 5.0 + 2.5
            sy = (j // 20) * 5.0 + 2.5

            # True destination
            w_gt = h20 * sx + h21 * sy + h22
            dx_gt = (h00 * sx + h01 * sy + h02) / w_gt
            dy_gt = (h10 * sx + h11 * sy + h12) / w_gt

            # Noisy observed destination
            noise_x = 0.4 * math.sin(j * 2.1 + trial * 0.3)
            noise_y = 0.4 * math.cos(j * 1.7 - trial * 0.5)
            dx_obs = dx_gt + noise_x
            dy_obs = dy_gt + noise_y

            # Predicted destination under trial model
            w_est = e20 * sx + e21 * sy + e22
            dx_pred = (e00 * sx + e01 * sy + e02) / w_est
            dy_pred = (e10 * sx + e11 * sy + e12) / w_est

            ex = dx_pred - dx_obs
            ey = dy_pred - dy_obs
            e2 = ex * ex + ey * ey
            err_sum += e2
            if e2 < threshold2:
                inliers += 1
            total_checks += 1

        if inliers > best_inliers:
            best_inliers = inliers
        total_err += err_sum / n_pts

    return (best_inliers, total_checks, total_err / n_iter)
