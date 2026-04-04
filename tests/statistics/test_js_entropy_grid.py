"""Test js_entropy_grid equivalence."""

import pytest

from cnake_charmer.cy.statistics.js_entropy_grid import js_entropy_grid as cy_func
from cnake_charmer.py.statistics.js_entropy_grid import js_entropy_grid as py_func


@pytest.mark.parametrize(
    "p_bias,q_bias,rows,cols,eps",
    [
        (0.01, 0.03, 40, 50, 1e-12),
        (0.03, 0.07, 80, 90, 1e-12),
        (0.09, 0.12, 70, 60, 1e-10),
    ],
)
def test_js_entropy_grid_equivalence(p_bias, q_bias, rows, cols, eps):
    py_result = py_func(p_bias, q_bias, rows, cols, eps)
    cy_result = cy_func(p_bias, q_bias, rows, cols, eps)
    for p, c in zip(py_result, cy_result, strict=False):
        assert abs(p - c) / max(abs(p), 1.0) < 1e-9
