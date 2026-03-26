"""Test 1D convolution equivalence."""

import pytest

from cnake_charmer.cy.numerical.convolution_1d import convolution_1d as cy_convolution_1d
from cnake_charmer.py.numerical.convolution_1d import convolution_1d as py_convolution_1d


@pytest.mark.parametrize("n", [10, 100, 1000, 10000])
def test_convolution_1d_equivalence(n):
    py_result = py_convolution_1d(n)
    cy_result = cy_convolution_1d(n)
    assert len(py_result) == len(cy_result), (
        f"Length mismatch: py={len(py_result)}, cy={len(cy_result)}"
    )
    for i in range(len(py_result)):
        assert abs(py_result[i] - cy_result[i]) < 1e-6, (
            f"Mismatch at index {i}: py={py_result[i]}, cy={cy_result[i]}"
        )
