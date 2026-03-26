"""Test cycle_detection equivalence."""

import pytest

from cnake_charmer.cy.graph.cycle_detection import cycle_detection as cy_func
from cnake_charmer.py.graph.cycle_detection import cycle_detection as py_func


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_cycle_detection_equivalence(n):
    assert py_func(n) == cy_func(n)
