"""Test union_tagged_value equivalence."""

import pytest

from cnake_charmer.cy.algorithms.union_tagged_value import (
    union_tagged_value as cy_func,
)
from cnake_charmer.py.algorithms.union_tagged_value import (
    union_tagged_value as py_func,
)


@pytest.mark.parametrize("n", [100, 1000, 10000, 100000])
def test_union_tagged_value_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert abs(py_result - cy_result) < 1e-6, f"Mismatch: py={py_result}, cy={cy_result}"
