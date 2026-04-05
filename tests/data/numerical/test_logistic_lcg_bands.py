"""Test logistic_lcg_bands equivalence."""

import pytest

from cnake_data.cy.numerical.logistic_lcg_bands import logistic_lcg_bands as cy_func
from cnake_data.py.numerical.logistic_lcg_bands import logistic_lcg_bands as py_func


@pytest.mark.parametrize("n", [10, 100, 1000, 10000])
def test_logistic_lcg_bands_equivalence(n):
    assert py_func(n) == cy_func(n)
