"""Test delta_decode_rows equivalence."""

import pytest

from cnake_data.cy.compression.delta_decode_rows import delta_decode_rows as cy_func
from cnake_data.py.compression.delta_decode_rows import delta_decode_rows as py_func


@pytest.mark.parametrize("n", [1, 10, 100, 1000])
def test_delta_decode_rows_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert py_result == cy_result, f"Mismatch at n={n}: {py_result} vs {cy_result}"
