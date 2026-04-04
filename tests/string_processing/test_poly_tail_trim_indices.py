"""Test poly_tail_trim_indices equivalence."""

import pytest

from cnake_charmer.cy.string_processing.poly_tail_trim_indices import (
    poly_tail_trim_indices as cy_func,
)
from cnake_charmer.py.string_processing.poly_tail_trim_indices import (
    poly_tail_trim_indices as py_func,
)


@pytest.mark.parametrize(
    "seq_count,seq_len,motif_shift",
    [
        (10, 24, 1),
        (120, 40, 3),
        (500, 64, 5),
        (1200, 72, 9),
    ],
)
def test_poly_tail_trim_indices_equivalence(seq_count, seq_len, motif_shift):
    assert py_func(seq_count, seq_len, motif_shift) == cy_func(seq_count, seq_len, motif_shift)
