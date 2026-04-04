"""Test matrix_transpose equivalence."""

import pytest

from cnake_data.cy.numerical.matrix_transpose import matrix_transpose as cy_func
from cnake_data.py.numerical.matrix_transpose import matrix_transpose as py_func


@pytest.mark.parametrize("n", [10, 50, 100, 200])
def test_matrix_transpose_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    for p, c in zip(py_result, cy_result, strict=False):
        assert abs(p - c) / max(abs(p), 1.0) < 1e-6, f"Mismatch: py={p}, cy={c}"
