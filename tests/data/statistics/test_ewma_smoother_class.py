"""Test ewma_smoother_class equivalence."""

import pytest

from cnake_data.cy.statistics.ewma_smoother_class import ewma_smoother_class as cy_func
from cnake_data.py.statistics.ewma_smoother_class import ewma_smoother_class as py_func


@pytest.mark.parametrize("alpha,steps,bias", [(0.2, 300, 0.1), (0.21, 500, 0.3), (0.35, 450, -0.1)])
def test_ewma_smoother_class_equivalence(alpha, steps, bias):
    py_result = py_func(alpha, steps, bias)
    cy_result = cy_func(alpha, steps, bias)
    for p, c in zip(py_result, cy_result, strict=False):
        assert abs(p - c) / max(abs(p), 1.0) < 1e-9
