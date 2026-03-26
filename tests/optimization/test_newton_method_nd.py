"""Test newton_method_nd equivalence."""

import pytest

from cnake_charmer.cy.optimization.newton_method_nd import newton_method_nd as cy_func
from cnake_charmer.py.optimization.newton_method_nd import newton_method_nd as py_func


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_newton_method_nd_equivalence(n):
    assert py_func(n) == cy_func(n)
