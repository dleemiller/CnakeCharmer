"""Test stack_invoice_digit_sum equivalence."""

import pytest

from cnake_data.cy.string_processing.stack_invoice_digit_sum import (
    stack_invoice_digit_sum as cy_func,
)
from cnake_data.py.string_processing.stack_invoice_digit_sum import (
    stack_invoice_digit_sum as py_func,
)


@pytest.mark.parametrize("n_fields", [50, 200, 1000, 4000])
def test_stack_invoice_digit_sum_equivalence(n_fields):
    assert py_func(n_fields) == cy_func(n_fields)
