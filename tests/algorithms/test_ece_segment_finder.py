"""Test ECE segment finder equivalence."""

import pytest

from cnake_charmer.cy.algorithms.ece_segment_finder import (
    ece_segment_finder as cy_ece_segment_finder,
)
from cnake_charmer.py.algorithms.ece_segment_finder import (
    ece_segment_finder as py_ece_segment_finder,
)


@pytest.mark.parametrize("n", [50, 100, 200, 500])
def test_ece_segment_finder_equivalence(n):
    assert py_ece_segment_finder(n) == cy_ece_segment_finder(n)
