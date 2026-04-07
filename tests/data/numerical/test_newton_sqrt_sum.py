"""Test newton_sqrt_sum equivalence."""

import pytest

from cnake_data.cy.numerical.newton_sqrt_sum import newton_sqrt_sum as cy_func
from cnake_data.py.numerical.newton_sqrt_sum import newton_sqrt_sum as py_func


@pytest.mark.parametrize("n", [10, 100, 1000, 5000])
def test_newton_sqrt_sum_equivalence(n):
    py_total, py_mid, py_last = py_func(n)
    cy_total, cy_mid, cy_last = cy_func(n)
    assert py_total == cy_total
    assert py_mid == cy_mid
    assert py_last == cy_last
