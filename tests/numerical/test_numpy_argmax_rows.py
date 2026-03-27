"""Test numpy_argmax_rows equivalence."""

import pytest

from cnake_charmer.cy.numerical.numpy_argmax_rows import (
    numpy_argmax_rows as cy_func,
)
from cnake_charmer.py.numerical.numpy_argmax_rows import (
    numpy_argmax_rows as py_func,
)


@pytest.mark.parametrize("n", [10, 100, 500])
def test_numpy_argmax_rows_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert py_result == cy_result, f"Mismatch: py={py_result}, cy={cy_result}"
