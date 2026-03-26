"""Test levenshtein_automaton equivalence."""

import pytest

from cnake_charmer.cy.algorithms.levenshtein_automaton import levenshtein_automaton as cy_func
from cnake_charmer.py.algorithms.levenshtein_automaton import levenshtein_automaton as py_func


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_levenshtein_automaton_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert py_result == cy_result, f"Mismatch: py={py_result}, cy={cy_result}"
