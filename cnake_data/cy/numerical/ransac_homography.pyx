# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""RANSAC robust homography inlier counting — Cython implementation."""

from libc.math cimport cos, sin

from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(200, 80))
def ransac_homography(int n_pts, int n_iter):
    """Run RANSAC inlier counting for a noisy homography correspondence set."""
    cdef double h00 = 1.02, h01 = 0.01, h02 = 8.0
    cdef double h10 = -0.005, h11 = 0.99, h12 = 5.0
    cdef double h20 = 0.0002, h21 = 0.00005, h22 = 1.0
    cdef double threshold2 = 4.0
    cdef int best_inliers = 0, total_checks = 0, inliers
    cdef double total_err = 0.0, err_sum
    cdef double scale, offset, e00, e01, e02, e10, e11, e12, e20, e21, e22
    cdef double sx, sy, w_gt, dx_gt, dy_gt, noise_x, noise_y, dx_obs, dy_obs
    cdef double w_est, dx_pred, dy_pred, ex, ey, e2
    cdef int trial, j

    for trial in range(n_iter):
        scale = 1.0 + 0.005 * sin(trial * 1.3)
        offset = 0.5 * cos(trial * 0.7)
        e00 = h00 * scale; e01 = h01; e02 = h02 + offset
        e10 = h10; e11 = h11 * scale; e12 = h12 - offset * 0.5
        e20 = h20; e21 = h21; e22 = h22
        inliers = 0
        err_sum = 0.0

        for j in range(n_pts):
            sx = (j % 20) * 5.0 + 2.5
            sy = (j // 20) * 5.0 + 2.5
            w_gt = h20 * sx + h21 * sy + h22
            dx_gt = (h00 * sx + h01 * sy + h02) / w_gt
            dy_gt = (h10 * sx + h11 * sy + h12) / w_gt
            noise_x = 0.4 * sin(j * 2.1 + trial * 0.3)
            noise_y = 0.4 * cos(j * 1.7 - trial * 0.5)
            dx_obs = dx_gt + noise_x
            dy_obs = dy_gt + noise_y
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
