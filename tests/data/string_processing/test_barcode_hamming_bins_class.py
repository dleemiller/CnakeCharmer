"""Test barcode_hamming_bins_class equivalence."""

import pytest

from cnake_data.cy.string_processing.barcode_hamming_bins_class import (
    barcode_hamming_bins_class as cy_func,
)
from cnake_data.py.string_processing.barcode_hamming_bins_class import (
    barcode_hamming_bins_class as py_func,
)


@pytest.mark.parametrize(
    "n_codes,code_len,edits,threshold,seed",
    [(30, 12, 2, 4, 7), (45, 16, 3, 6, 13), (60, 14, 2, 5, 21)],
)
def test_barcode_hamming_bins_class_equivalence(n_codes, code_len, edits, threshold, seed):
    assert py_func(n_codes, code_len, edits, threshold, seed) == cy_func(
        n_codes, code_len, edits, threshold, seed
    )
