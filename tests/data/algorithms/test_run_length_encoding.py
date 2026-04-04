"""Test run_length_encoding equivalence."""

import pytest

from cnake_data.cy.algorithms.run_length_encoding import run_length_encoding as cy_func
from cnake_data.py.algorithms.run_length_encoding import run_length_encoding as py_func


@pytest.mark.parametrize("n", [0, 1, 10, 100, 500, 1000])
def test_run_length_encoding_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert py_result == cy_result
