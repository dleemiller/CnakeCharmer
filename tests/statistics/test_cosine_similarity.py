"""Test cosine_similarity equivalence."""

import pytest

from cnake_charmer.cy.statistics.cosine_similarity import cosine_similarity as cy_func
from cnake_charmer.py.statistics.cosine_similarity import cosine_similarity as py_func


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_cosine_similarity_equivalence(n):
    py_total, py_last = py_func(n)
    cy_total, cy_last = cy_func(n)
    assert abs(py_total - cy_total) / max(abs(py_total), 1.0) < 1e-6
    assert abs(py_last - cy_last) / max(abs(py_last), 1.0) < 1e-6
