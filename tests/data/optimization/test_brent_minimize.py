"""Test brent_minimize equivalence."""

import pytest

from cnake_data.cy.optimization.brent_minimize import brent_minimize as cy_func
from cnake_data.py.optimization.brent_minimize import brent_minimize as py_func


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_brent_minimize_equivalence(n):
    assert abs(py_func(n) - cy_func(n)) < 1e-6
