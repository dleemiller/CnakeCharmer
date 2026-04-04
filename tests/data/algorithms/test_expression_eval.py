"""Test expression_eval equivalence."""

import pytest

from cnake_data.cy.algorithms.expression_eval import expression_eval as cy_func
from cnake_data.py.algorithms.expression_eval import expression_eval as py_func


@pytest.mark.parametrize("n", [100, 500, 1000, 5000])
def test_expression_eval_equivalence(n):
    assert abs(py_func(n) - cy_func(n)) < 1e-4
