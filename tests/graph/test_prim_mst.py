"""Test prim_mst equivalence."""

import pytest

from cnake_charmer.cy.graph.prim_mst import prim_mst as cy_func
from cnake_charmer.py.graph.prim_mst import prim_mst as py_func


@pytest.mark.parametrize("n", [50, 200, 500, 1000])
def test_prim_mst_equivalence(n):
    assert py_func(n) == cy_func(n)
