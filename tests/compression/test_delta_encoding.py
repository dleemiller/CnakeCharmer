"""Test delta_encoding equivalence."""

import pytest

from cnake_charmer.cy.compression.delta_encoding import delta_encoding as cy_func
from cnake_charmer.py.compression.delta_encoding import delta_encoding as py_func


@pytest.mark.parametrize("n", [10, 100, 1000, 10000])
def test_delta_encoding_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert py_result == cy_result, f"Mismatch at n={n}: {py_result} vs {cy_result}"
