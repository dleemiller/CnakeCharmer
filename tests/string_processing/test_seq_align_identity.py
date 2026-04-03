"""Test seq_align_identity equivalence."""

import pytest

from cnake_charmer.cy.string_processing.seq_align_identity import (
    seq_align_identity as cy_func,
)
from cnake_charmer.py.string_processing.seq_align_identity import (
    seq_align_identity as py_func,
)


@pytest.mark.parametrize("n,gap_rate", [(100, 10), (500, 15), (1000, 20)])
def test_seq_align_identity_equivalence(n, gap_rate):
    py_result = py_func(n, gap_rate)
    cy_result = cy_func(n, gap_rate)
    for p, c in zip(py_result, cy_result, strict=False):
        assert abs(p - c) / max(abs(p), 1.0) < 1e-6
