"""Polar sampling helpers for 2D Fourier power images."""

from __future__ import annotations

import numpy as np


def recentre_image(img, forward=True):
    shift0 = img.shape[0] // 2
    shift1 = img.shape[1] // 2
    if forward:
        img = np.roll(img, shift0, axis=0)
        img = np.roll(img, shift1, axis=1)
        return img
    img = np.roll(-img, shift0, axis=0)
    img = np.roll(-img, shift1, axis=1)
    return img


def get_sample_radii(rad1, rad2, nrad):
    root_radii = np.linspace(np.sqrt(rad1), np.sqrt(rad2), nrad + 1)
    radii = 0.5 * (root_radii[:-1] + root_radii[1:])
    return radii * radii


def get_sample_rotations(npolar_rows, nsamples):
    bounds = np.linspace(0.0, 2.0 * np.pi / npolar_rows, nsamples + 1)
    return 0.5 * (bounds[:-1] + bounds[1:])
