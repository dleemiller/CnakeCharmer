"""Test qr_decomposition equivalence."""

import pytest

from cnake_data.cy.numerical.qr_decomposition import qr_decomposition as cy_func
from cnake_data.py.numerical.qr_decomposition import qr_decomposition as py_func


@pytest.mark.parametrize("n", [10, 50, 100, 150])
def test_qr_decomposition_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    for p, c in zip(py_result, cy_result, strict=False):
        assert abs(p - c) / max(abs(p), 1.0) < 1e-6, f"Mismatch: py={p}, cy={c}"
