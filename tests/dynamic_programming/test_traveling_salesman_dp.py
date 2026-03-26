"""Test traveling_salesman_dp equivalence."""

import pytest

from cnake_charmer.cy.dynamic_programming.traveling_salesman_dp import (
    traveling_salesman_dp as cy_func,
)
from cnake_charmer.py.dynamic_programming.traveling_salesman_dp import (
    traveling_salesman_dp as py_func,
)


@pytest.mark.parametrize("n", [1, 4, 8, 12, 15])
def test_traveling_salesman_dp_equivalence(n):
    assert py_func(n) == cy_func(n)
