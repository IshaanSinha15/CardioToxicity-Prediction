"""Dose-response modeling utilities for cardiac ion-channel block prediction."""

from .channel_block_generator import ChannelBlockGenerator, ChannelIC50Inputs, ChannelBlockResult
from .concentration_profiles import (
    DEFAULT_MULTIPLES,
    ConcentrationProfile,
    calculate_free_concentration,
    calculate_total_concentration,
    categorize_exposure,
    generate_concentration_series,
)
from .dose_response_curve import (
    DoseResponseCurve,
    generate_curve,
    plot_channel_comparison,
    plot_dose_response_curves,
    plot_safety_margin_bars,
    plot_therapeutic_vs_toxic_dose,
)
from .hill_equation import (
    HillEquation,
    HillFitResult,
    HillParameters,
    calculate_block_with_uncertainty,
    fit_hill_parameters,
    hill_block_percentage,
    validate_hill_parameters,
)
from .safety_margin import (
    SafetyMarginAnalyzer,
    SafetyMarginResult,
    calculate_safety_margin,
    classify_safety_margin,
)
from .validation import (
    ValidationIssue,
    ValidationReport,
    validate_block_curve,
    validate_channel_inputs,
    validate_hill_coefficient,
    validate_ic50_range,
    validate_ord_payload,
)
