import pytest

from cnake_data.cy.string_processing.stack2_kmer_profile import stack2_kmer_profile as cy_func
from cnake_data.py.string_processing.stack2_kmer_profile import stack2_kmer_profile as py_func


@pytest.mark.parametrize(
    "dna_length, motif_width, seed_tag",
    [(4000, 4, 3), (9000, 5, 11), (14000, 6, 23)],
)
def test_stack2_kmer_profile_equivalence(dna_length, motif_width, seed_tag):
    assert py_func(dna_length, motif_width, seed_tag) == cy_func(dna_length, motif_width, seed_tag)
