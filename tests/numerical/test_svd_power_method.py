"""Test svd_power_method equivalence."""

import pytest

from cnake_charmer.cy.numerical.svd_power_method import svd_power_method as cy_func
from cnake_charmer.py.numerical.svd_power_method import svd_power_method as py_func


@pytest.mark.parametrize("n", [10, 30, 50, 100])
def test_svd_power_method_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    for p, c in zip(py_result, cy_result, strict=False):
        assert abs(p - c) / max(abs(p), 1.0) < 1e-4, f"Mismatch: py={py_result}, cy={cy_result}"
