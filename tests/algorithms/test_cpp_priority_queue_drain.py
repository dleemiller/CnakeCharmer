"""Test cpp_priority_queue_drain equivalence between Python and Cython."""

import pytest

from cnake_charmer.cy.algorithms.cpp_priority_queue_drain import cpp_priority_queue_drain as cy_func
from cnake_charmer.py.algorithms.cpp_priority_queue_drain import cpp_priority_queue_drain as py_func


@pytest.mark.parametrize("n", [1000, 50000, 500000])
def test_cpp_priority_queue_drain_equivalence(n):
    py_r = py_func(n)
    cy_r = cy_func(n)
    assert py_r == cy_r
