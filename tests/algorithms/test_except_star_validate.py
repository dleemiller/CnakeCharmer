"""Test except_star_validate equivalence."""

import pytest

from cnake_charmer.cy.algorithms.except_star_validate import (
    except_star_validate as cy_func,
)
from cnake_charmer.py.algorithms.except_star_validate import (
    except_star_validate as py_func,
)


@pytest.mark.parametrize("n", [10, 100, 1000, 50000])
def test_except_star_validate_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert py_result == cy_result, f"Mismatch: py={py_result}, cy={cy_result}"
