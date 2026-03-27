"""Test numpy_softmax equivalence."""

import pytest

from cnake_charmer.cy.nn_ops.numpy_softmax import (
    numpy_softmax as cy_func,
)
from cnake_charmer.py.nn_ops.numpy_softmax import (
    numpy_softmax as py_func,
)


@pytest.mark.parametrize("n", [256, 2560, 25600])
def test_numpy_softmax_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    tol = max(1e-3, abs(py_result) * 1e-6)
    assert abs(py_result - cy_result) < tol, f"Mismatch: py={py_result}, cy={cy_result}"
