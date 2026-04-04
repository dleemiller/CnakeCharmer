"""Test fibonacci_iterator equivalence."""

import pytest

from cnake_data.cy.math_problems.fibonacci_iterator import fibonacci_iterator as cy_func
from cnake_data.py.math_problems.fibonacci_iterator import fibonacci_iterator as py_func


@pytest.mark.parametrize("n", [10, 100, 1000, 10000])
def test_fibonacci_iterator_equivalence(n):
    assert py_func(n) == cy_func(n)
