"""Test dateline_bbox equivalence."""

import pytest

from cnake_charmer.cy.geometry.dateline_bbox import dateline_bbox as cy_func
from cnake_charmer.py.geometry.dateline_bbox import dateline_bbox as py_func


@pytest.mark.parametrize("n", [100, 1000, 5000, 15000])
def test_dateline_bbox_equivalence(n):
    assert py_func(n) == cy_func(n)
