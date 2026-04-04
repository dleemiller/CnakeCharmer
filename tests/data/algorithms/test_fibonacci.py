"""Test fibonacci equivalence."""

import pytest

from cnake_data.cy.algorithms.fibonacci import fibonacci as cy_fibonacci
from cnake_data.py.algorithms.fibonacci import fibonacci as py_fibonacci


@pytest.mark.parametrize("n", [10, 100, 1000, 10000])
def test_fibonacci_equivalence(n):
    assert py_fibonacci(n) == cy_fibonacci(n)
