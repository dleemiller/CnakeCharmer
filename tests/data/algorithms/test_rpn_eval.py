"""Test rpn_eval equivalence."""

import pytest

from cnake_data.cy.algorithms.rpn_eval import rpn_eval as cy_func
from cnake_data.py.algorithms.rpn_eval import rpn_eval as py_func


@pytest.mark.parametrize("n", [100, 1000, 5000, 10000])
def test_rpn_eval_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert abs(py_result - cy_result) < max(1e-3, abs(py_result) * 1e-6), (
        f"Mismatch: py={py_result}, cy={cy_result}"
    )
