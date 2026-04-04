"""Test egg_drop equivalence."""

import pytest

from cnake_data.cy.dynamic_programming.egg_drop import egg_drop as cy_func
from cnake_data.py.dynamic_programming.egg_drop import egg_drop as py_func


@pytest.mark.parametrize("n", [1, 10, 100, 1000, 5000])
def test_egg_drop_equivalence(n):
    assert py_func(n) == cy_func(n)
