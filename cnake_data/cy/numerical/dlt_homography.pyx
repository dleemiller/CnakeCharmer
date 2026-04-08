# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Reprojection error evaluation for homography correspondences — Cython implementation."""

from libc.math cimport cos, pi, sin, sqrt

from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(1000,))
def dlt_homography(int n_pts):
    """Evaluate reprojection errors for noisy homography correspondences."""
    cdef double h00 = 1.05, h01 = 0.02, h02 = 8.0
    cdef double h10 = -0.01, h11 = 0.98, h12 = 5.0
    cdef double h20 = 0.0001, h21 = 0.00005, h22 = 1.0

    cdef double e00 = h00 + 0.01, e01 = h01, e02 = h02 + 1.2
    cdef double e10 = h10, e11 = h11 - 0.005, e12 = h12 + 0.8
    cdef double e20 = h20, e21 = h21, e22 = h22

    cdef double sum_err = 0.0, max_err = 0.0
    cdef int n_inliers = 0
    cdef double threshold = 2.0
    cdef double t, sx, sy, w_gt, dx_gt, dy_gt
    cdef double noise_x, noise_y, dx_obs, dy_obs
    cdef double w_est, dx_pred, dy_pred, ex, ey, err
    cdef int i

    for i in range(n_pts):
        t = 2.0 * pi * i / n_pts
        sx = 50.0 + 40.0 * cos(t)
        sy = 50.0 + 40.0 * sin(t)

        w_gt = h20 * sx + h21 * sy + h22
        dx_gt = (h00 * sx + h01 * sy + h02) / w_gt
        dy_gt = (h10 * sx + h11 * sy + h12) / w_gt

        noise_x = 0.3 * sin(i * 1.7)
        noise_y = 0.3 * cos(i * 2.3)
        dx_obs = dx_gt + noise_x
        dy_obs = dy_gt + noise_y

        w_est = e20 * sx + e21 * sy + e22
        dx_pred = (e00 * sx + e01 * sy + e02) / w_est
        dy_pred = (e10 * sx + e11 * sy + e12) / w_est

        ex = dx_pred - dx_obs
        ey = dy_pred - dy_obs
        err = sqrt(ex * ex + ey * ey)
        sum_err += err
        if err > max_err:
            max_err = err
        if err < threshold:
            n_inliers += 1

    return (sum_err, max_err, n_inliers)
