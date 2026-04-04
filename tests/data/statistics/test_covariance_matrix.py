"""Test covariance_matrix equivalence."""

import pytest

from cnake_data.cy.statistics.covariance_matrix import (
    covariance_matrix as cy_covariance_matrix,
)
from cnake_data.py.statistics.covariance_matrix import (
    covariance_matrix as py_covariance_matrix,
)


@pytest.mark.parametrize("n", [10, 100, 1000, 10000])
def test_covariance_matrix_equivalence(n):
    py_result = py_covariance_matrix(n)
    cy_result = cy_covariance_matrix(n)
    assert abs(py_result - cy_result) < 1e-6, f"Mismatch: py={py_result}, cy={cy_result}"
