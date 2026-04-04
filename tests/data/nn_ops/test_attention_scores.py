"""Test attention_scores equivalence."""

import pytest

from cnake_data.cy.nn_ops.attention_scores import attention_scores as cy_func
from cnake_data.py.nn_ops.attention_scores import attention_scores as py_func


@pytest.mark.parametrize("n", [10, 50, 100, 200])
def test_attention_scores_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert abs(py_result - cy_result) < 1e-3, f"Mismatch: py={py_result}, cy={cy_result}"
