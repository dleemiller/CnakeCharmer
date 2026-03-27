"""Test prefix sum equivalence."""

import pytest

from cnake_charmer.cy.numerical.contig_prefix_sum import (
    contig_prefix_sum as cy_contig_prefix_sum,
)
from cnake_charmer.py.numerical.contig_prefix_sum import (
    contig_prefix_sum as py_contig_prefix_sum,
)


@pytest.mark.parametrize("n", [100, 1000, 10000, 100000])
def test_contig_prefix_sum_equivalence(n):
    py_result = py_contig_prefix_sum(n)
    cy_result = cy_contig_prefix_sum(n)
    assert abs(py_result - cy_result) < 1e-3, f"Mismatch: py={py_result}, cy={cy_result}"
