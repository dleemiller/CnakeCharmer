# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Perspective camera projection and reprojection error — Cython implementation."""

from libc.math cimport cos, sin, sqrt

from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(1000,))
def camera_projection(int n_pts):
    """Project n_pts 3D calibration target points and compute reprojection error."""
    cdef double fx = 800.0, fy = 800.0, cx = 320.0, cy = 240.0

    # Rotation from Euler angles ax=0.02, ay=0.05, az=0.10
    cdef double ax = 0.02, ay = 0.05, az = 0.10
    cdef double cax = cos(ax), sax = sin(ax)
    cdef double cay = cos(ay), say = sin(ay)
    cdef double caz = cos(az), saz = sin(az)
    cdef double r00 = cay * caz
    cdef double r01 = caz * sax * say - cax * saz
    cdef double r02 = cax * caz * say + sax * saz
    cdef double r10 = cay * saz
    cdef double r11 = cax * caz + sax * say * saz
    cdef double r12 = cax * say * saz - caz * sax
    cdef double r20 = -say
    cdef double r21 = cay * sax
    cdef double r22 = cax * cay
    cdef double tx = 0.3, ty = -0.2, tz = 5.0

    cdef double pfx = fx + 2.0, pfy = fy - 1.5
    cdef double pcx = cx + 0.5, pcy = cy - 0.3

    cdef double sum_err = 0.0, max_err = 0.0
    cdef int n_inliers = 0
    cdef double threshold = 1.5
    cdef double X, Y, Z, xc, yc, zc
    cdef double u_gt, v_gt, u_obs, v_obs, u_pred, v_pred
    cdef double eu, ev, err
    cdef int i, col_i, row_i

    for i in range(n_pts):
        col_i = i % 40
        row_i = i // 40
        X = (col_i - 20) * 0.05
        Y = (row_i - 12) * 0.05
        Z = 0.02 * sin(i * 0.1)

        xc = r00 * X + r01 * Y + r02 * Z + tx
        yc = r10 * X + r11 * Y + r12 * Z + ty
        zc = r20 * X + r21 * Y + r22 * Z + tz

        u_gt = fx * xc / zc + cx
        v_gt = fy * yc / zc + cy
        u_obs = u_gt + 0.5 * sin(i * 1.3)
        v_obs = v_gt + 0.5 * cos(i * 1.7)

        u_pred = pfx * xc / zc + pcx
        v_pred = pfy * yc / zc + pcy

        eu = u_pred - u_obs
        ev = v_pred - v_obs
        err = sqrt(eu * eu + ev * ev)
        sum_err += err
        if err > max_err:
            max_err = err
        if err < threshold:
            n_inliers += 1

    return (sum_err, max_err, n_inliers)
