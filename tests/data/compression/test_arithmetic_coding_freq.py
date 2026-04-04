"""Test arithmetic_coding_freq equivalence."""

import pytest

from cnake_data.cy.compression.arithmetic_coding_freq import arithmetic_coding_freq as cy_func
from cnake_data.py.compression.arithmetic_coding_freq import arithmetic_coding_freq as py_func


@pytest.mark.parametrize("n", [100, 1000, 10000, 50000])
def test_arithmetic_coding_freq_equivalence(n):
    assert abs(py_func(n) - cy_func(n)) < 1e-6
