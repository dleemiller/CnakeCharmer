"""Test run-length encode equivalence."""

import pytest

from cnake_charmer.cy.string_processing.run_length_encode import (
    run_length_encode as cy_run_length_encode,
)
from cnake_charmer.py.string_processing.run_length_encode import (
    run_length_encode as py_run_length_encode,
)


@pytest.mark.parametrize("n", [10, 100, 1000, 10000])
def test_run_length_encode_equivalence(n):
    assert py_run_length_encode(n) == cy_run_length_encode(n)
