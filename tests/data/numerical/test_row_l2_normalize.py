"""Test row_l2_normalize equivalence."""

import pytest

from cnake_data.cy.numerical.row_l2_normalize import row_l2_normalize as cy_func
from cnake_data.py.numerical.row_l2_normalize import row_l2_normalize as py_func


@pytest.mark.parametrize(
    "n,m",
    [(5, 4), (20, 10), (100, 50), (300, 200)],
)
def test_row_l2_normalize_equivalence(n, m):
    assert py_func(n, m) == cy_func(n, m)
