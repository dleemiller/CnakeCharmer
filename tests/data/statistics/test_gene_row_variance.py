"""Test gene_row_variance equivalence."""

import pytest

from cnake_data.cy.statistics.gene_row_variance import gene_row_variance as cy_func
from cnake_data.py.statistics.gene_row_variance import gene_row_variance as py_func


@pytest.mark.parametrize("n", [10, 100, 500, 2000])
def test_gene_row_variance_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    for a, b in zip(py_result, cy_result, strict=False):
        assert abs(a - b) / max(abs(a), 1.0) < 1e-4
