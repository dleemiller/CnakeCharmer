"""Test batch_norm equivalence."""

import pytest

from cnake_charmer.cy.nn_ops.batch_norm import batch_norm as cy_func
from cnake_charmer.py.nn_ops.batch_norm import batch_norm as py_func


@pytest.mark.parametrize("n", [100, 1000, 10000, 100000])
def test_batch_norm_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert abs(py_result - cy_result) < 1e-3, f"Mismatch: py={py_result}, cy={cy_result}"
