import numpy as np

from classification_backend.dose_response.hill_equation import (
    HillEquation,
    fit_hill_parameters,
    hill_block_percentage,
    validate_hill_parameters,
)


def test_hill_block_percentage_is_50_percent_at_ic50():
    assert hill_block_percentage(100.0, 100.0, 1.0) == 50.0


def test_hill_equation_is_monotonic():
    equation = HillEquation(ic50_nm=100.0, hill_coefficient=1.0)
    blocks = equation.block_series(np.array([1.0, 10.0, 100.0, 1000.0]))

    assert np.all(np.diff(blocks) > 0)
    assert blocks[0] < blocks[-1]


def test_fit_hill_parameters_recovers_simple_curve():
    concentrations = np.array([10.0, 30.0, 100.0, 300.0, 1000.0])
    responses = hill_block_percentage(concentrations, 100.0, 1.0)

    result = fit_hill_parameters(concentrations, responses)

    assert abs(result.ic50_nm - 100.0) < 1e-6
    assert abs(result.hill_coefficient - 1.0) < 1e-6


def test_validate_hill_parameters_returns_warnings_for_unusual_values():
    is_valid, warnings = validate_hill_parameters(100.0, 2.5, channel="hERG")

    assert is_valid is True
    assert warnings
