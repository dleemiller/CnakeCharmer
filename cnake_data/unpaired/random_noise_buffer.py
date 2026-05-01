"""Generate Gaussian random noise values."""

from __future__ import annotations

import random


def random_noise(number=1):
    ran = random.normalvariate
    return [ran(0, 1) for _ in range(number)]
