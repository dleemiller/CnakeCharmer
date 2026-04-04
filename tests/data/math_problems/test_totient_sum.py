"""Test totient_sum equivalence."""

import pytest

from cnake_data.cy.math_problems.totient_sum import totient_sum as cy_func
from cnake_data.py.math_problems.totient_sum import totient_sum as py_func


@pytest.mark.parametrize("n", [1, 10, 100, 1000, 10000])
def test_totient_sum_equivalence(n):
    assert py_func(n) == cy_func(n)
