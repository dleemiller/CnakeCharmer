"""Perspective camera projection and reprojection error.

Projects 3D points through a pinhole camera model (intrinsics K and extrinsics
[R|t]), adds deterministic noise, and evaluates errors under a perturbed model.
Core evaluation step in iterative camera calibration algorithms.

Keywords: camera projection, pinhole model, reprojection error, camera calibration, geometric vision
"""

import math

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(1000,))
def camera_projection(n_pts: int) -> tuple:
    """Project n_pts 3D calibration target points and compute reprojection error.

    Uses a known rotation (small Euler angles) and translation. Evaluates
    reprojection error for a slightly perturbed intrinsic model.

    Args:
        n_pts: Number of 3D-2D point correspondences.

    Returns:
        Tuple of (sum_error, max_error, n_inliers) with threshold=1.5 px.
    """
    # Camera intrinsics (true)
    fx, fy = 800.0, 800.0
    cx, cy = 320.0, 240.0

    # Rotation matrix from Euler angles ax=0.02, ay=0.05, az=0.10
    ax, ay, az = 0.02, 0.05, 0.10
    cax = math.cos(ax)
    sax = math.sin(ax)
    cay = math.cos(ay)
    say = math.sin(ay)
    caz = math.cos(az)
    saz = math.sin(az)
    r00 = cay * caz
    r01 = caz * sax * say - cax * saz
    r02 = cax * caz * say + sax * saz
    r10 = cay * saz
    r11 = cax * caz + sax * say * saz
    r12 = cax * say * saz - caz * sax
    r20 = -say
    r21 = cay * sax
    r22 = cax * cay
    tx, ty, tz = 0.3, -0.2, 5.0

    # Perturbed intrinsics for evaluation
    pfx, pfy = fx + 2.0, fy - 1.5
    pcx, pcy = cx + 0.5, cy - 0.3

    sum_err = 0.0
    max_err = 0.0
    n_inliers = 0
    threshold = 1.5

    for i in range(n_pts):
        # 3D points on a planar grid with slight depth variation
        col_i = i % 40
        row_i = i // 40
        X = (col_i - 20) * 0.05
        Y = (row_i - 12) * 0.05
        Z = 0.02 * math.sin(i * 0.1)

        # Apply extrinsics (R, t) once — shared by both projections
        xc = r00 * X + r01 * Y + r02 * Z + tx
        yc = r10 * X + r11 * Y + r12 * Z + ty
        zc = r20 * X + r21 * Y + r22 * Z + tz

        # True projection + noise
        u_gt = fx * xc / zc + cx
        v_gt = fy * yc / zc + cy
        u_obs = u_gt + 0.5 * math.sin(i * 1.3)
        v_obs = v_gt + 0.5 * math.cos(i * 1.7)

        # Perturbed projection
        u_pred = pfx * xc / zc + pcx
        v_pred = pfy * yc / zc + pcy

        eu = u_pred - u_obs
        ev = v_pred - v_obs
        err = math.sqrt(eu * eu + ev * ev)
        sum_err += err
        if err > max_err:
            max_err = err
        if err < threshold:
            n_inliers += 1

    return (sum_err, max_err, n_inliers)
