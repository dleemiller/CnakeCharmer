"""Test pairwise_l1_norm equivalence."""

import pytest

from cnake_charmer.cy.numerical.pairwise_l1_norm import pairwise_l1_norm as cy_func
from cnake_charmer.py.numerical.pairwise_l1_norm import pairwise_l1_norm as py_func


@pytest.mark.parametrize("n1,n2,k", [(5, 5, 3), (20, 20, 5), (50, 30, 8)])
def test_pairwise_l1_norm_equivalence(n1, n2, k):
    py_result = py_func(n1, n2, k)
    cy_result = cy_func(n1, n2, k)
    for p, c in zip(py_result, cy_result, strict=False):
        assert abs(p - c) / max(abs(p), 1.0) < 1e-6
