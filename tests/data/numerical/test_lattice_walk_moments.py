"""Test lattice_walk_moments equivalence."""

import pytest

from cnake_data.cy.numerical.lattice_walk_moments import lattice_walk_moments as cy_func
from cnake_data.py.numerical.lattice_walk_moments import lattice_walk_moments as py_func


@pytest.mark.parametrize("n", [10, 100, 1000, 5000])
def test_lattice_walk_moments_equivalence(n):
    assert py_func(n) == cy_func(n)
