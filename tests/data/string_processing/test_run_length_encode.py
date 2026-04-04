"""Test run_length_encode equivalence."""

import pytest

from cnake_data.cy.string_processing.run_length_encode import run_length_encode as cy_func
from cnake_data.py.string_processing.run_length_encode import run_length_encode as py_func


@pytest.mark.parametrize("n", [10, 100, 1000, 10000])
def test_run_length_encode_equivalence(n):
    assert py_func(n) == cy_func(n)
