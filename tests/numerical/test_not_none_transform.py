"""Test not_none_transform equivalence."""

import pytest

from cnake_charmer.cy.numerical.not_none_transform import not_none_transform as cy_func
from cnake_charmer.py.numerical.not_none_transform import not_none_transform as py_func


@pytest.mark.parametrize("n", [10, 100, 1000, 10000])
def test_not_none_transform_equivalence(n):
    assert abs(py_func(n) - cy_func(n)) < 1e-6
