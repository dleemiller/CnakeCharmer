"""Test babylonian_sqrt equivalence."""

import pytest

from cnake_charmer.cy.numerical.babylonian_sqrt import babylonian_sqrt_sum as cy_babylonian_sqrt_sum
from cnake_charmer.py.numerical.babylonian_sqrt import babylonian_sqrt_sum as py_babylonian_sqrt_sum


@pytest.mark.parametrize("n", [1, 10, 100, 1000])
def test_babylonian_sqrt_sum_equivalence(n):
    assert abs(py_babylonian_sqrt_sum(n) - cy_babylonian_sqrt_sum(n)) < 1e-6
