"""Test spearman_rank equivalence."""

import pytest

from cnake_charmer.cy.statistics.spearman_rank import spearman_rank as cy_func
from cnake_charmer.py.statistics.spearman_rank import spearman_rank as py_func


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_spearman_rank_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    for a, b in zip(py_result, cy_result, strict=False):
        assert abs(a - b) / max(abs(a), 1.0) < 1e-4
