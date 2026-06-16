import numpy as np
import pandas as pd

from classification_backend.dose_response import (
    ChannelBlockGenerator,
    ChannelIC50Inputs,
    ConcentrationProfile,
    HillEquation,
    calculate_safety_margin,
    classify_safety_margin,
    generate_concentration_series,
    hill_block_percentage,
    plot_dose_response_curves,
)


def test_hill_block_percentage_is_50_percent_at_ic50():
    assert hill_block_percentage(100.0, 100.0, 1.0) == 50.0


def test_hill_equation_is_monotonic():
    equation = HillEquation(ic50_nm=100.0, hill_coefficient=1.0)
    blocks = equation.block_series(np.array([1.0, 10.0, 100.0, 1000.0]))
    assert np.all(np.diff(blocks) > 0)


def test_concentration_profile_generates_requested_multiples():
    profile = ConcentrationProfile(reference_concentration_nm=10.0)
    assert list(profile.multiples) == [0.01, 0.1, 1.0, 10.0, 100.0]
    assert profile.concentrations_nm.tolist() == [0.1, 1.0, 10.0, 100.0, 1000.0]


def test_channel_block_generator_returns_ord_payload_shape():
    generator = ChannelBlockGenerator(
        ChannelIC50Inputs(herg_ic50_nm=100.0, nav_ic50_nm=200.0, cav_ic50_nm=300.0)
    )
    payload = generator.to_ord_payload(100.0)
    assert set(payload.keys()) == {"concentration", "herg_block", "nav_block", "cav_block"}
    assert payload["concentration"] == 100.0
    assert payload["herg_block"] == 50.0


def test_channel_block_generator_profile_has_five_levels():
    generator = ChannelBlockGenerator(
        ChannelIC50Inputs(herg_ic50_nm=100.0, nav_ic50_nm=200.0, cav_ic50_nm=300.0)
    )
    frame = generator.block_profile(reference_concentration_nm=10.0)
    assert len(frame) == 5
    assert frame["concentration"].tolist() == [0.1, 1.0, 10.0, 100.0, 1000.0]


def test_safety_margin_thresholds():
    margin = calculate_safety_margin(1000.0, 10.0)
    assert margin == 100.0
    assert classify_safety_margin(margin) == "Safe"


def test_generate_concentration_series_matches_profile():
    concentrations = generate_concentration_series(10.0)
    assert concentrations.tolist() == [0.1, 1.0, 10.0, 100.0, 1000.0]


def test_plot_function_creates_figure():
    curves = {
        "hERG": {"ic50_nm": 100.0, "hill_coefficient": 1.0},
        "Nav1.5": {"ic50_nm": 200.0, "hill_coefficient": 1.0},
        "Cav1.2": {"ic50_nm": 300.0, "hill_coefficient": 1.0},
    }
    fig, ax = plot_dose_response_curves(curves, reference_concentration_nm=10.0)
    assert fig is not None
    assert ax is not None
