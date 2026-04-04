"""Test not_none_matrix_sum equivalence."""

import pytest

from cnake_data.cy.numerical.not_none_matrix_sum import not_none_matrix_sum as cy_func
from cnake_data.py.numerical.not_none_matrix_sum import not_none_matrix_sum as py_func


@pytest.mark.parametrize("n", [10, 100, 1000, 5000])
def test_not_none_matrix_sum_equivalence(n):
    assert abs(py_func(n) - cy_func(n)) < 1e-2
