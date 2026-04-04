"""Test fasta_lcg_gc_counts equivalence."""

import pytest

from cnake_charmer.cy.string_processing.fasta_lcg_gc_counts import (
    fasta_lcg_gc_counts as cy_func,
)
from cnake_charmer.py.string_processing.fasta_lcg_gc_counts import (
    fasta_lcg_gc_counts as py_func,
)


@pytest.mark.parametrize(
    "seed,ia,ic,im,length,gc_threshold",
    [
        (42, 3877, 29573, 139968, 500, 0.5),
        (11, 1664525, 1013904223, 2147483647, 1200, 0.8),
        (99, 1103515245, 12345, 2147483648, 900, 0.7),
    ],
)
def test_fasta_lcg_gc_counts_equivalence(seed, ia, ic, im, length, gc_threshold):
    assert py_func(seed, ia, ic, im, length, gc_threshold) == cy_func(
        seed,
        ia,
        ic,
        im,
        length,
        gc_threshold,
    )
