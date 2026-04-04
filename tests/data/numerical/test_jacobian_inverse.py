"""Test jacobian_inverse equivalence."""

import pytest

from cnake_data.cy.numerical.jacobian_inverse import jacobian_inverse as cy_func
from cnake_data.py.numerical.jacobian_inverse import jacobian_inverse as py_func


@pytest.mark.parametrize("n", [50, 200, 500, 2000])
def test_jacobian_inverse_equivalence(n):
    assert py_func(n) == cy_func(n)
