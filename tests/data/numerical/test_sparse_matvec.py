"""Test sparse_matvec equivalence between Python and Cython."""

import pytest

from cnake_data.cy.numerical.sparse_matvec import sparse_matvec as cy_sparse_matvec
from cnake_data.py.numerical.sparse_matvec import sparse_matvec as py_sparse_matvec


@pytest.mark.parametrize("n", [100, 1000, 10000, 100000])
def test_sparse_matvec_equivalence(n):
    assert py_sparse_matvec(n) == cy_sparse_matvec(n)
