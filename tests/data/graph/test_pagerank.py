"""Test pagerank equivalence."""

import pytest

from cnake_data.cy.graph.pagerank import pagerank as cy_func
from cnake_data.py.graph.pagerank import pagerank as py_func


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_pagerank_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    for p, c in zip(py_result, cy_result, strict=False):
        assert abs(p - c) < 1e-6
