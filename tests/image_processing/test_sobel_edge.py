"""Test sobel_edge equivalence."""

import pytest

from cnake_charmer.cy.image_processing.sobel_edge import sobel_edge as cy_sobel_edge
from cnake_charmer.py.image_processing.sobel_edge import sobel_edge as py_sobel_edge


@pytest.mark.parametrize("n", [10, 50, 100, 200])
def test_sobel_edge_equivalence(n):
    py_result = py_sobel_edge(n)
    cy_result = cy_sobel_edge(n)
    assert py_result == cy_result, f"Mismatch: py={py_result}, cy={cy_result}"
