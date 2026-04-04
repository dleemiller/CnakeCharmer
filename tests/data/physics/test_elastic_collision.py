"""Test elastic_collision equivalence."""

import pytest

from cnake_data.cy.physics.elastic_collision import elastic_collision as cy_func
from cnake_data.py.physics.elastic_collision import elastic_collision as py_func


@pytest.mark.parametrize("n_digits,v_init", [(2, 1.0), (3, 1.0), (4, 0.5)])
def test_elastic_collision_equivalence(n_digits, v_init):
    py_result = py_func(n_digits, v_init)
    cy_result = cy_func(n_digits, v_init)
    for p, c in zip(py_result, cy_result, strict=False):
        assert abs(p - c) / max(abs(p), 1.0) < 1e-6
