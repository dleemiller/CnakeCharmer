"""Test final_accumulator equivalence."""

import pytest

from cnake_charmer.cy.numerical.final_accumulator import final_accumulator as cy_func
from cnake_charmer.py.numerical.final_accumulator import final_accumulator as py_func


@pytest.mark.parametrize("n", [100, 1000, 10000, 100000])
def test_final_accumulator_equivalence(n):
    assert abs(py_func(n) - cy_func(n)) < 1e-4
