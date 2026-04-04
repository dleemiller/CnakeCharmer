"""Test newton_raphson_poly equivalence."""

import pytest

from cnake_data.cy.numerical.newton_raphson_poly import (
    newton_raphson_poly as cy_newton_raphson_poly,
)
from cnake_data.py.numerical.newton_raphson_poly import (
    newton_raphson_poly as py_newton_raphson_poly,
)


@pytest.mark.parametrize("n", [10, 100, 1000, 10000])
def test_newton_raphson_poly_equivalence(n):
    py_result = py_newton_raphson_poly(n)
    cy_result = cy_newton_raphson_poly(n)

    py_checksum, py_count = py_result
    cy_checksum, cy_count = cy_result

    # Float tolerance for checksum
    rel_err = abs(py_checksum - cy_checksum) / max(abs(py_checksum), 1.0)
    assert rel_err < 1e-4, (
        f"Checksum mismatch: py={py_checksum}, cy={cy_checksum}, rel_err={rel_err}"
    )

    # Converge count should match exactly
    assert py_count == cy_count, f"Converge count mismatch: py={py_count}, cy={cy_count}"
