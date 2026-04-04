"""Test canonical_kmer_count equivalence."""

import pytest

from cnake_data.cy.string_processing.canonical_kmer_count import canonical_kmer_count as cy_func
from cnake_data.py.string_processing.canonical_kmer_count import canonical_kmer_count as py_func


@pytest.mark.parametrize(
    "seq_len,k,stride",
    [
        (30, 3, 2),
        (120, 5, 7),
        (1000, 9, 3),
        (5000, 11, 7),
    ],
)
def test_canonical_kmer_count_equivalence(seq_len, k, stride):
    assert py_func(seq_len, k, stride) == cy_func(seq_len, k, stride)
