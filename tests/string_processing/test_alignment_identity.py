"""Test alignment_identity equivalence."""

import pytest

from cnake_charmer.cy.string_processing.alignment_identity import alignment_identity as cy_func
from cnake_charmer.py.string_processing.alignment_identity import alignment_identity as py_func


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_alignment_identity_equivalence(n):
    py_total, py_high = py_func(n)
    cy_total, cy_high = cy_func(n)
    assert abs(py_total - cy_total) / max(abs(py_total), 1.0) < 1e-6
    assert py_high == cy_high
