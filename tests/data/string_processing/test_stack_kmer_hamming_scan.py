"""Test stack_kmer_hamming_scan equivalence."""

import pytest

from cnake_data.cy.string_processing.stack_kmer_hamming_scan import (
    stack_kmer_hamming_scan as cy_func,
)
from cnake_data.py.string_processing.stack_kmer_hamming_scan import (
    stack_kmer_hamming_scan as py_func,
)


@pytest.mark.parametrize("args", [(64, 5, 1), (120, 7, 2), (240, 9, 2), (400, 11, 3)])
def test_stack_kmer_hamming_scan_equivalence(args):
    assert py_func(*args) == cy_func(*args)
