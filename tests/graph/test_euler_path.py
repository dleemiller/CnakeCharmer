"""Test euler_path equivalence."""

import pytest

from cnake_charmer.cy.graph.euler_path import euler_path as cy_func
from cnake_charmer.py.graph.euler_path import euler_path as py_func


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_euler_path_equivalence(n):
    assert py_func(n) == cy_func(n)
