"""Test fibonacci_matrix equivalence."""

import pytest

from cnake_data.cy.math_problems.fibonacci_matrix import fibonacci_matrix as cy_func
from cnake_data.py.math_problems.fibonacci_matrix import fibonacci_matrix as py_func


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_fibonacci_matrix_equivalence(n):
    assert py_func(n) == cy_func(n)
