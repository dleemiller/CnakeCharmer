"""Test cppclass_vec2d_ops equivalence between Python and Cython."""

import pytest

from cnake_data.cy.geometry.cppclass_vec2d_ops import cppclass_vec2d_ops as cy_func
from cnake_data.py.geometry.cppclass_vec2d_ops import cppclass_vec2d_ops as py_func


@pytest.mark.parametrize("n", [1000, 30000, 300000])
def test_cppclass_vec2d_ops_equivalence(n):
    py_r = py_func(n)
    cy_r = cy_func(n)
    assert abs(py_r[0] - cy_r[0]) / max(abs(py_r[0]), 1.0) < 1e-6
    assert abs(py_r[1] - cy_r[1]) / max(abs(py_r[1]), 1.0) < 1e-6
