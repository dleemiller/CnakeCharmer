"""Test stable_marriage equivalence."""

import pytest

from cnake_charmer.cy.algorithms.stable_marriage import stable_marriage as cy_func
from cnake_charmer.py.algorithms.stable_marriage import stable_marriage as py_func


@pytest.mark.parametrize("n", [10, 50, 100, 200])
def test_stable_marriage_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert py_result == cy_result
