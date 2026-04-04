"""Test dutch_national_flag equivalence."""

import pytest

from cnake_data.cy.algorithms.dutch_national_flag import dutch_national_flag as cy_func
from cnake_data.py.algorithms.dutch_national_flag import dutch_national_flag as py_func


@pytest.mark.parametrize("n", [10, 100, 1000, 10000])
def test_dutch_national_flag_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert py_result == cy_result, f"Mismatch at n={n}"
