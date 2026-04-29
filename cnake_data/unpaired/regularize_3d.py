"""3D EPR, total variation, and Tikhonov regularization terms with gradients."""

from __future__ import annotations

import numpy as np


def epr(img, fwd=False):
    cimg = np.asarray(img, dtype=np.float64)
    nx, ny, nz = cimg.shape
    if min(nx, ny, nz) < 3:
        raise ValueError("Minimum size of img is (3, 3, 3)")

    gimg = np.zeros((nx, ny, nz), dtype=np.float64)
    eprnorm = 0.0
    fdir = 1 if not fwd else -1

    for i in range(1, nx - 1):
        im = i - fdir
        for j in range(1, ny - 1):
            jm = j - fdir
            for k in range(1, nz - 1):
                km = k - fdir
                cv = cimg[i, j, k]
                dxf = cv - cimg[im, j, k]
                dyf = cv - cimg[i, jm, k]
                dzf = cv - cimg[i, j, km]
                dxxf = dxf * dxf
                dyyf = dyf * dyf
                dzzf = dzf * dzf

                gnorm = dxxf / (dxxf + 1.0) + dyyf / (dyyf + 1.0) + dzzf / (dzzf + 1.0)
                eprnorm += gnorm

                dxxf += 1.0
                dyyf += 1.0
                dzzf += 1.0

                gimg[i, j, k] += 2 * dxf / (dxxf * dxxf)
                gimg[i, j, k] += 2 * dyf / (dyyf * dyyf)
                gimg[i, j, k] += 2 * dzf / (dzzf * dzzf)

                gimg[im, j, k] -= 2 * dxf / (dxxf * dxxf)
                gimg[i, jm, k] -= 2 * dyf / (dyyf * dyyf)
                gimg[i, j, km] -= 2 * dzf / (dzzf * dzzf)

    return eprnorm, gimg


def totvar(img, fwd=False, eps=0.01):
    cimg = np.asarray(img, dtype=np.float64)
    nx, ny, nz = cimg.shape
    if min(nx, ny, nz) < 3:
        raise ValueError("Minimum size of img is (3, 3, 3)")

    gimg = np.zeros((nx, ny, nz), dtype=np.float64)
    tvnorm = 0.0
    heps = 0.5 * eps
    fdir = 1 if not fwd else -1

    for i in range(1, nx - 1):
        im = i - fdir
        for j in range(1, ny - 1):
            jm = j - fdir
            for k in range(1, nz - 1):
                km = k - fdir
                cv = cimg[i, j, k]
                dxf = cv - cimg[im, j, k]
                dyf = cv - cimg[i, jm, k]
                dzf = cv - cimg[i, j, km]
                gnorm = np.sqrt(dxf * dxf + dyf * dyf + dzf * dzf)
                tvnorm += gnorm
                gnorm += heps
                gimg[i, j, k] += (dxf + dyf + dzf) / gnorm
                gimg[im, j, k] -= dxf / gnorm
                gimg[i, jm, k] -= dyf / gnorm
                gimg[i, j, km] -= dzf / gnorm

    return tvnorm, gimg


def tikhonov(img):
    cimg = np.asarray(img, dtype=np.float64)
    nx, ny, nz = cimg.shape
    if min(nx, ny, nz) < 3:
        raise ValueError("Minimum image size is (3, 3, 3)")
    tnorm = 0.5 * np.sum(cimg * cimg)
    return float(tnorm), cimg
