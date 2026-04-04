"""Test chebyshev_nodes equivalence."""

import pytest

from cnake_data.cy.numerical.chebyshev_nodes import chebyshev_nodes as cy_func
from cnake_data.py.numerical.chebyshev_nodes import chebyshev_nodes as py_func


@pytest.mark.parametrize("n", [10, 100, 1000, 5000])
def test_chebyshev_nodes_equivalence(n):
    assert abs(py_func(n) - cy_func(n)) < 1e-6
