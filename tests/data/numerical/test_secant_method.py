"""Test secant_method equivalence."""

import pytest

from cnake_data.cy.numerical.secant_method import secant_method as cy_func
from cnake_data.py.numerical.secant_method import secant_method as py_func


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_secant_method_equivalence(n):
    assert abs(py_func(n) - cy_func(n)) < 1e-6
