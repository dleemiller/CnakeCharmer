"""Simple flood-risk economic cost helpers."""

from __future__ import annotations

import numpy as np


def cost_fun(
    ratio: float, c: float, b: float, lambd: float, dikeinit: float, dikeincrease: float
) -> float:
    dikeincrease_cm = dikeincrease * 100.0
    dikeinit_cm = dikeinit * 100.0
    cost = ((c + b * dikeincrease_cm) * np.exp(lambd * (dikeinit_cm + dikeincrease_cm))) * ratio
    return float(cost * 1e6)


def discount(amount: float, rate: float, n: int) -> float:
    factor = 1.0 + rate / 100.0
    periods = np.arange(1, n + 1)
    disc_amount = amount * 1.0 / (factor**periods)
    return float(np.sum(disc_amount))


def cost_evacuation(n_evacuated: int, days_to_threat: int) -> float:
    return float(n_evacuated * 22 * (days_to_threat + 3) * int(days_to_threat > 0))
