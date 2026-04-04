"""Test tarjan_scc equivalence."""

import pytest

from cnake_data.cy.algorithms.tarjan_scc import tarjan_scc as cy_func
from cnake_data.py.algorithms.tarjan_scc import tarjan_scc as py_func


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_tarjan_scc_equivalence(n):
    assert py_func(n) == cy_func(n)
