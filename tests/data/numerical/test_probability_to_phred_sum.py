"""Test probability_to_phred_sum equivalence."""

import pytest

from cnake_data.cy.numerical.probability_to_phred_sum import probability_to_phred_sum as cy_func
from cnake_data.py.numerical.probability_to_phred_sum import probability_to_phred_sum as py_func


@pytest.mark.parametrize(
    "seed,samples,floor",
    [(48271, 100, 1e-6), (7, 5000, 1e-5), (12345, 20000, 1e-4)],
)
def test_probability_to_phred_sum_equivalence(seed, samples, floor):
    py_result = py_func(seed, samples, floor)
    cy_result = cy_func(seed, samples, floor)
    assert abs(py_result - cy_result) / max(abs(py_result), 1.0) < 1e-10
