"""Test nogil_matrix_multiply equivalence."""

import pytest

from cnake_data.cy.numerical.nogil_matrix_multiply import (
    nogil_matrix_multiply as cy_func,
)
from cnake_data.py.numerical.nogil_matrix_multiply import (
    nogil_matrix_multiply as py_func,
)


@pytest.mark.parametrize("n", [2, 5, 10, 50])
def test_nogil_matrix_multiply_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    rel = abs(py_result - cy_result) / max(abs(py_result), 1.0)
    assert rel < 1e-6, f"Mismatch: py={py_result}, cy={cy_result}"
