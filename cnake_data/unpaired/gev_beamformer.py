"""Generalized eigenvector beamforming helpers."""

from __future__ import annotations

import numpy as np


def gev_beamformer_vectors(target_psd, noise_psd):
    """Compute principal generalized eigenvectors per frequency bin."""
    bins = target_psd.shape[0]
    sensors = target_psd.shape[1]
    out = np.empty((bins, sensors), dtype=np.complex128)

    for f in range(bins):
        evals, evecs = np.linalg.eig(np.linalg.solve(noise_psd[f], target_psd[f]))
        idx = int(np.argmax(np.abs(evals)))
        out[f] = evecs[:, idx]
    return out
