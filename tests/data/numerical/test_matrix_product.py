"""Test matrix_product equivalence between Python and Cython."""

import pytest

from cnake_data.cy.numerical.matrix_product import matrix_product as cy_matrix_product
from cnake_data.py.numerical.matrix_product import matrix_product as py_matrix_product


@pytest.mark.parametrize("n", [100, 1000, 50000, 500000])
def test_matrix_product_equivalence(n):
    assert py_matrix_product(n) == cy_matrix_product(n)
