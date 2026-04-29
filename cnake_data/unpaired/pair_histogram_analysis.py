"""Pairwise orientation-position histogram analysis with periodic boundaries."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HistAxis:
    n_bins: int
    min: float
    max: float

    def __post_init__(self) -> None:
        if self.n_bins < 1:
            raise ValueError("n_bins must be >= 1")
        if self.min >= self.max:
            raise ValueError("min must be < max")

    @property
    def bin_width(self) -> float:
        return (self.max - self.min) / self.n_bins

    def index(self, x: float) -> int:
        return int(math.floor((x - self.min) / self.bin_width))


class Hist2DData:
    def __init__(
        self, a1: HistAxis, a2: HistAxis | None = None, count=None, acc_value=None
    ) -> None:
        self.a1 = a1
        self.a2 = a2 or a1
        shape = (self.a1.n_bins, self.a2.n_bins)
        self.count = np.zeros(shape, dtype=np.uint64) if count is None else np.asarray(count)
        self.acc_value = (
            np.zeros(shape, dtype=np.float64) if acc_value is None else np.asarray(acc_value)
        )

    def add_value(self, x1: float, x2: float, value: float) -> None:
        i1 = self.a1.index(x1)
        i2 = self.a2.index(x2)
        if 0 <= i1 < self.a1.n_bins and 0 <= i2 < self.a2.n_bins:
            self.count[i1, i2] += 1
            self.acc_value[i1, i2] += value


def periodic_direction(p1: np.ndarray, p2: np.ndarray, box_size: float) -> np.ndarray:
    d = p2 - p1
    return d - box_size * np.round(d / box_size)


def analyze_configuration(
    positions: np.ndarray,
    orientations: np.ndarray,
    box_size: float,
    hist_cartesian: Hist2DData,
    hist_angle: Hist2DData,
) -> None:
    if hist_angle.a1.min != 0:
        raise ValueError("hist_angle.a1.min must be 0")

    r_max = hist_angle.a1.max
    pi_over_2 = math.pi / 2.0

    for i in range(positions.shape[0]):
        p_i = positions[i]
        o_i = orientations[i]
        for j in range(positions.shape[0]):
            if i == j:
                continue
            r = periodic_direction(p_i, positions[j], box_size)
            r2 = float(np.dot(r, r))
            ra = float(np.dot(o_i, r))
            re_sq = max(0.0, r2 - ra * ra)
            re = math.sqrt(re_sq)

            if abs(ra) > r_max or re > r_max:
                continue

            alpha = float(np.dot(o_i, orientations[j]))
            hist_cartesian.add_value(re, ra, alpha)

            r_length = math.sqrt(r2)
            if r_length <= r_max and r_length > 0:
                theta = pi_over_2 - math.acos(max(-1.0, min(1.0, ra / r_length)))
                hist_angle.add_value(r_length, theta, alpha)
