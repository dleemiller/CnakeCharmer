"""Test tarjan_bridges equivalence."""

import pytest

from cnake_charmer.cy.graph.tarjan_bridges import tarjan_bridges as cy_func
from cnake_charmer.py.graph.tarjan_bridges import tarjan_bridges as py_func


@pytest.mark.parametrize("n", [10, 100, 1000, 10000])
def test_tarjan_bridges_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert py_result == cy_result, f"Mismatch at n={n}: {py_result} vs {cy_result}"
