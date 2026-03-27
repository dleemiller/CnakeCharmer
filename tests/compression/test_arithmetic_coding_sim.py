"""Test arithmetic_coding_sim equivalence."""

import pytest

from cnake_charmer.cy.compression.arithmetic_coding_sim import arithmetic_coding_sim as cy_func
from cnake_charmer.py.compression.arithmetic_coding_sim import arithmetic_coding_sim as py_func


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_arithmetic_coding_sim_equivalence(n):
    assert py_func(n) == cy_func(n)
