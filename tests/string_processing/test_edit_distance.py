"""Test edit distance equivalence."""

import pytest

from cnake_charmer.cy.string_processing.edit_distance import edit_distance as cy_edit_distance
from cnake_charmer.py.string_processing.edit_distance import edit_distance as py_edit_distance


@pytest.mark.parametrize("n", [5, 20, 50, 100])
def test_edit_distance_equivalence(n):
    assert py_edit_distance(n) == cy_edit_distance(n)
