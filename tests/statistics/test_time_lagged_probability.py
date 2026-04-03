"""Test time_lagged_probability equivalence."""

import pytest

from cnake_charmer.cy.statistics.time_lagged_probability import time_lagged_probability as cy_func
from cnake_charmer.py.statistics.time_lagged_probability import time_lagged_probability as py_func


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_time_lagged_probability_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    for p, c in zip(py_result, cy_result, strict=False):
        assert abs(p - c) / max(abs(p), 1.0) < 1e-4, f"Mismatch: py={py_result}, cy={cy_result}"
