"""Test softmax equivalence."""

import pytest

from cnake_charmer.cy.numerical.softmax import softmax as cy_softmax
from cnake_charmer.py.numerical.softmax import softmax as py_softmax


@pytest.mark.parametrize("n", [100, 1000, 10000])
def test_softmax_equivalence(n):
    py_result = py_softmax(n)
    cy_result = cy_softmax(n)
    assert abs(py_result - cy_result) < 1e-6, f"Mismatch: py={py_result}, cy={cy_result}"
