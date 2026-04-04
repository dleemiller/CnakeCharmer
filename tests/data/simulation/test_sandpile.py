"""Test sandpile equivalence."""

import pytest

from cnake_data.cy.simulation.sandpile import sandpile as cy_func
from cnake_data.py.simulation.sandpile import sandpile as py_func


@pytest.mark.parametrize("n", [10, 30, 50])
def test_sandpile_equivalence(n):
    assert py_func(n) == cy_func(n)
