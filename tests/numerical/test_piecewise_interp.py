"""Test piecewise_interp equivalence."""

import pytest

from cnake_charmer.cy.numerical.piecewise_interp import piecewise_interp as cy_func
from cnake_charmer.py.numerical.piecewise_interp import piecewise_interp as py_func


@pytest.mark.parametrize("n", [100, 500, 1000, 5000])
def test_piecewise_interp_equivalence(n):
    assert py_func(n) == cy_func(n)
