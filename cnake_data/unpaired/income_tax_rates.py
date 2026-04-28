"""Income tax rate helper kernels."""

from __future__ import annotations

import numpy as np


def get_income_tax_rate(salary: int) -> float:
    if salary <= 11850:
        return 0.0
    if salary <= 46350:
        return 0.2
    if salary <= 150000:
        return 0.4
    return 0.45


def repeat_income_tax_rates(salary: int, n_repeats: int):
    out = np.zeros(n_repeats, dtype=float)
    rate = get_income_tax_rate(salary)
    for i in range(n_repeats):
        out[i] = rate
    return out
