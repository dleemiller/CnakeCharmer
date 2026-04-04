"""Test kmer_frequency equivalence."""

import pytest

from cnake_charmer.cy.string_processing.kmer_frequency import kmer_frequency as cy_func
from cnake_charmer.py.string_processing.kmer_frequency import kmer_frequency as py_func


@pytest.mark.parametrize("n", [1000, 10000, 100000, 500000])
def test_kmer_frequency_equivalence(n):
    assert py_func(n) == cy_func(n)
