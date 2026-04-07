"""Test row_l2_normalize equivalence."""

import pytest

from cnake_data.cy.numerical.row_l2_normalize import row_l2_normalize as cy_func
from cnake_data.py.numerical.row_l2_normalize import row_l2_normalize as py_func


@pytest.mark.parametrize(
    "n,m",
    [(5, 4), (20, 10), (100, 50), (300, 200)],
)
def test_row_l2_normalize_equivalence(n, m):
    py_total, py_first, py_last = py_func(n, m)
    cy_total, cy_first, cy_last = cy_func(n, m)
    assert abs(py_total - cy_total) / max(abs(py_total), 1.0) < 1e-9
    assert abs(py_first - cy_first) < 1e-12
    assert abs(py_last - cy_last) < 1e-12
