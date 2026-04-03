"""Test dna_kmer_canonical equivalence."""

import pytest

from cnake_charmer.cy.string_processing.dna_kmer_canonical import dna_kmer_canonical as cy_func
from cnake_charmer.py.string_processing.dna_kmer_canonical import dna_kmer_canonical as py_func


@pytest.mark.parametrize("n", [20, 100, 1000, 10000])
def test_dna_kmer_canonical_equivalence(n):
    assert py_func(n) == cy_func(n)
