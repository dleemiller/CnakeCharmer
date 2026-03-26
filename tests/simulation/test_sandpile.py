"""Test sandpile equivalence."""

import pytest

from cnake_charmer.cy.simulation.sandpile import sandpile as cy_func
from cnake_charmer.py.simulation.sandpile import sandpile as py_func


@pytest.mark.parametrize("n", [10, 30, 50])
def test_sandpile_equivalence(n):
    assert py_func(n) == cy_func(n)
