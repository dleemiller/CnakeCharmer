"""Test mtf_encode equivalence."""

import pytest

from cnake_charmer.cy.compression.mtf_encode import mtf_encode as cy_func
from cnake_charmer.py.compression.mtf_encode import mtf_encode as py_func


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_mtf_encode_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert py_result == cy_result, f"Mismatch at n={n}: {py_result} vs {cy_result}"
