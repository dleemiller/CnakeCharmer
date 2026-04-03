"""Test kmer_frequency equivalence."""

import pytest

from cnake_charmer.cy.string_processing.kmer_frequency import kmer_frequency as cy_func
from cnake_charmer.py.string_processing.kmer_frequency import kmer_frequency as py_func


@pytest.mark.parametrize("n", [100, 500, 2000])
def test_kmer_frequency_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert py_result == cy_result
