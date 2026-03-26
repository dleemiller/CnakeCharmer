"""Test conjugate_gradient equivalence."""

import pytest

from cnake_charmer.cy.optimization.conjugate_gradient import conjugate_gradient as cy_func
from cnake_charmer.py.optimization.conjugate_gradient import conjugate_gradient as py_func


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_conjugate_gradient_equivalence(n):
    assert abs(py_func(n) - cy_func(n)) < 1e-6
