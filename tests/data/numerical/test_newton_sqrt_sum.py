"""Test newton_sqrt_sum equivalence."""

import pytest

from cnake_data.cy.numerical.newton_sqrt_sum import newton_sqrt_sum as cy_func
from cnake_data.py.numerical.newton_sqrt_sum import newton_sqrt_sum as py_func


@pytest.mark.parametrize("n", [10, 100, 1000, 5000])
def test_newton_sqrt_sum_equivalence(n):
    assert py_func(n) == cy_func(n)
