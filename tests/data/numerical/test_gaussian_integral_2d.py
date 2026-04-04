"""Test gaussian_integral_2d equivalence."""

import pytest

from cnake_data.cy.numerical.gaussian_integral_2d import gaussian_integral_2d as cy_func
from cnake_data.py.numerical.gaussian_integral_2d import gaussian_integral_2d as py_func


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_gaussian_integral_2d_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    for p, c in zip(py_result, cy_result, strict=False):
        assert abs(p - c) / max(abs(p), 1.0) < 1e-4, f"Mismatch: py={py_result}, cy={cy_result}"
