"""Test kadane_2d equivalence."""

import pytest

from cnake_data.cy.algorithms.kadane_2d import kadane_2d as cy_func
from cnake_data.py.algorithms.kadane_2d import kadane_2d as py_func


@pytest.mark.parametrize("n", [5, 10, 20, 50])
def test_kadane_2d_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert py_result == cy_result
