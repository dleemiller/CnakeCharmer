"""Test quickselect equivalence."""

import pytest

from cnake_charmer.cy.algorithms.quickselect import quickselect as cy_quickselect
from cnake_charmer.py.algorithms.quickselect import quickselect as py_quickselect


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_quickselect_equivalence(n):
    assert py_quickselect(n) == cy_quickselect(n)
