"""Test threshold count equivalence."""

import pytest

from cnake_charmer.cy.image_processing.contig_threshold_count import (
    contig_threshold_count as cy_contig_threshold_count,
)
from cnake_charmer.py.image_processing.contig_threshold_count import (
    contig_threshold_count as py_contig_threshold_count,
)


@pytest.mark.parametrize("n", [100, 1000, 100000, 500000])
def test_contig_threshold_count_equivalence(n):
    py_result = py_contig_threshold_count(n)
    cy_result = cy_contig_threshold_count(n)
    assert py_result == cy_result, f"Mismatch: py={py_result}, cy={cy_result}"
