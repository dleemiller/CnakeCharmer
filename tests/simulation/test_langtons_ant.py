"""Test langtons_ant equivalence."""

import pytest

from cnake_charmer.cy.simulation.langtons_ant import langtons_ant as cy_func
from cnake_charmer.py.simulation.langtons_ant import langtons_ant as py_func


@pytest.mark.parametrize("n", [10, 50, 100, 200])
def test_langtons_ant_equivalence(n):
    assert py_func(n) == cy_func(n)
