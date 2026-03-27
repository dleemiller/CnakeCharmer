"""Test callable_transform equivalence."""

import pytest

from cnake_charmer.cy.numerical.callable_transform import callable_transform as cy_func
from cnake_charmer.py.numerical.callable_transform import callable_transform as py_func


@pytest.mark.parametrize("n", [100, 1000, 10000, 50000])
def test_callable_transform_equivalence(n):
    assert abs(py_func(n) - cy_func(n)) < 1e-6
