"""Test sequence_identity equivalence."""

import pytest

from cnake_charmer.cy.string_processing.sequence_identity import sequence_identity as cy_func
from cnake_charmer.py.string_processing.sequence_identity import sequence_identity as py_func


@pytest.mark.parametrize("n", [100, 500, 2000, 5000])
def test_sequence_identity_equivalence(n):
    assert py_func(n) == cy_func(n)
